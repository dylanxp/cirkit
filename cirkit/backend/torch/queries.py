import functools
from abc import ABC
from collections.abc import Sequence

import torch
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import TorchInnerLayer, TorchInputLayer, TorchLayer
from cirkit.utils.scope import Scope


class Query(ABC):
    """An object used to run queries of circuits compiled using the torch backend."""

    def __init__(self) -> None: ...


class IntegrateQuery(Query):
    """The integration query object allows marginalising out variables.

    Computes output in two forward passes:
        a) The normal circuit forward pass for input x
        b) The integration forward pass where all variables are marginalised

    A mask over random variables is computed based on the scopes passed as
    input. This determines whether the integrated or normal circuit result
    is returned for each variable.
    """

    def __init__(self, circuit: TorchCircuit) -> None:
        """Initialize an integration query object.

        Args:
            circuit: The circuit to integrate over.

        Raises:
            ValueError: If the circuit to integrate is not smooth or not decomposable.
        """
        if not circuit.properties.smooth or not circuit.properties.decomposable:
            raise ValueError(
                f"The circuit to integrate must be smooth and decomposable, "
                f"but found {circuit.properties}"
            )
        super().__init__()
        self._circuit = circuit

    def __call__(self, x: Tensor, *, integrate_vars: Tensor | Scope | Sequence[Scope]) -> Tensor:
        """Solve an integration query, given an input batch and the variables to integrate.

        Args:
            x: An input batch of shape $(B, D)$, where $B$ is the batch size,
                and $D$ is the number of variables.
            integrate_vars: The variables to integrate. It must be a subset of the variables on
                which the circuit given in the constructor is defined on.
                The format can be one of the following three:
                    1. Tensor of shape (B, D) where B is the batch size and D is the number of
                        variables in the scope of the circuit. Its dtype should be torch.bool
                        and have True in the positions of random variables that should be
                        marginalised out and False elsewhere.
                    2. Scope, in this case the same integration mask is applied for all entries
                        of the batch
                    3. Sequence of Scopes, where the length of the list must be either 1 or B. If
                        the list has length 1, behaves as above.
        Returns:
            The result of the integration query, given as a tensor of shape $(B, O, K)$,
                where $B$ is the batch size, $O$ is the number of output vectors of the circuit, and
                $K$ is the number of units in each output vector.
        """
        if isinstance(integrate_vars, Tensor):
            # Check type of tensor is boolean
            if integrate_vars.dtype != torch.bool:
                raise ValueError(
                    f"Expected dtype of tensor to be torch.bool, got {integrate_vars.dtype}"
                )
            # If single dimensional tensor, assume batch size = 1
            if len(integrate_vars.shape) == 1:
                integrate_vars = torch.unsqueeze(integrate_vars, 0)
            # If the scope is correct, proceed, otherwise error
            num_vars = max(self._circuit.scope) + 1
            if integrate_vars.shape[1] == num_vars:
                integrate_vars_mask = integrate_vars
            else:
                raise ValueError(
                    f"Circuit scope has {num_vars} variables but integrate_vars "
                    f"was defined over {integrate_vars.shape[1]} != {num_vars} variables"
                )
        else:
            # Convert list of scopes to a boolean mask of dimension (B, N) where
            # N is the number of variables in the circuit's scope.
            integrate_vars_mask = IntegrateQuery.scopes_to_mask(self._circuit, integrate_vars)
            integrate_vars_mask = integrate_vars_mask.to(x.device)

        # Check batch sizes of input x and mask are compatible
        if integrate_vars_mask.shape[0] not in (1, x.shape[0]):
            raise ValueError(
                "The number of scopes to integrate over must "
                "either match the batch size of x, or be 1 if you "
                "want to broadcast. Found #inputs = "
                f"{x.shape[0]} != {integrate_vars_mask.shape[0]} = len(integrate_vars)"
            )

        output = self._circuit.evaluate(
            x,
            module_fn=functools.partial(
                IntegrateQuery._layer_fn, integrate_vars_mask=integrate_vars_mask
            ),
        )  # (O, B, K)
        return output.transpose(0, 1)  # (B, O, K)

    @staticmethod
    def _layer_fn(layer: TorchLayer, x: Tensor, *, integrate_vars_mask: Tensor) -> Tensor:
        # Evaluate a layer: if it is not an input layer, then evaluate it in the usual
        # feed-forward way. Otherwise, use the variables to integrate to solve the marginal
        # queries on the input layers.
        output = layer(x)  # (F, B, Ko)
        if not isinstance(layer, TorchInputLayer):
            return output
        if layer.num_variables > 1:
            raise NotImplementedError("Integration of multivariate input layers is not supported")
        # Some information:
        # - integrate_vars_mask is a boolean tensor of dim (B, N)
        #   where N is the number of variables in the scope of the whole circuit.
        # - layer.scope_idx contains a subset of the variable_idxs of the scope
        #   but may be a reshaped tensor; the shape and order of the variables may be different.
        # As such, we need to use the idxs in layer.scope_idx to look-up the values from
        # the integrate_vars_mask. This will return the correct shape and values.
        # Note that, if integrate_vars_mask was a vector, we could do
        # integrate_vars_mask[layer.scope_idx] the vmap below applies the above across
        # the batch (B) dimension.

        # integration_mask has dimension (B, F, Ko)
        integration_mask = torch.vmap(lambda x: x[layer.scope_idx])(integrate_vars_mask)
        # permute to match integration_output: integration_mask has dimension (F, B, Ko)
        integration_mask = integration_mask.permute([1, 0, 2])
        if not torch.any(integration_mask).item():
            return output
        integration_output = layer.integrate()
        # Use the integration mask to select which output should be the result of
        # an integration operation, and which should not be
        # This is done in parallel for all folds, and regardless of whether the
        # circuit is folded or unfolded
        return torch.where(integration_mask, integration_output, output)

    @staticmethod
    def scopes_to_mask(
        circuit: TorchCircuit, batch_integrate_vars: Scope | Sequence[Scope]
    ) -> Tensor:
        """Accepts a batch of scopes and returns a boolean mask as a tensor with
        True in positions of specified scope indices and False otherwise.
        """
        # If we passed a single scope, assume B = 1
        if isinstance(batch_integrate_vars, Scope):
            batch_integrate_vars = [batch_integrate_vars]

        batch_size = len(batch_integrate_vars)
        # There are cases where the circuit.scope may change,
        # e.g. we may marginalise out X_1 and the length of the scope may be smaller
        # but the actual scope will not have been shifted.
        num_rvs = max(circuit.scope) + 1
        num_idxs = sum(len(s) for s in batch_integrate_vars)

        # TODO: Maybe consider using a sparse tensor
        mask = torch.zeros((batch_size, num_rvs), dtype=torch.bool)

        # Catch case of only empty scopes where the following command will fail
        if num_idxs == 0:
            return mask

        batch_idxs, rv_idxs = zip(
            *((i, idx) for i, idxs in enumerate(batch_integrate_vars) for idx in idxs if idxs)
        )

        # Check that we have not asked to marginalise variables that are not defined
        invalid_idxs = Scope(rv_idxs) - circuit.scope
        if invalid_idxs:
            raise ValueError(
                "The variables to marginalize must be a subset of "
                "the circuit scope. Invalid variables "
                f"not in scope: {list(invalid_idxs)} "
            )

        mask[batch_idxs, rv_idxs] = True
        return mask


class SamplingQuery(Query):
    """Sample from the circuit, optionally using evidence."""

    def __init__(self, circuit: TorchCircuit) -> None:
        """Initialize a sampling query object.

        Args:
            circuit: The circuit to sample from.

        Raises:
            ValueError: If the circuit is not smooth or not decomposable.
        """
        if not circuit.properties.smooth or not circuit.properties.decomposable:
            raise ValueError(
                f"MAP is supported by smooth and decomposable circuits, "
                f"found {circuit.properties}"
            )
        super().__init__()
        self._circuit = circuit

    def __call__(
        self, *, num_samples: int = 1, x: Tensor | None = None, evidence_vars: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Sample from the circuit, optionally using an input evidence.

        Args:
            x: The input evidence of shape $(B, D)$, where $B$ is the batch size,
                and $D$ is the number of variables.
            evidence_vars: The variables to include in the evidence. It must be a subset of the
                variables on which the circuit given in the constructor is defined on.
                It must be a Tensor of shape (B, D) where B is the batch size and D is the number of
                variables in the scope of the circuit. Its dtype should be torch.bool and have True
                in the positions of random variables that are in the evidence and False elsewhere.
        Returns:
            The result of the sampling query, given as a tuple where the first element is the probability of
            the sample value and the second value is the sample with shape $(B, D)$.
        """
        if (x is None) ^ (evidence_vars is None):
            assert ValueError("Both evidence and the evidence variables must be provided.")

        # >>> Commented out until determining why this check is here
        # TODO: understand why this is here
        # if self._circuit.symbolic_operation:
        #     circuit_scope = self._circuit.symbolic_operation.operands[0].scope
        # else:
        #     circuit_scope = self._circuit.scope
        # <<<
        circuit_scope = self._circuit.scope

        if x is None:
            # if the circuit is the result of some operation then work on the original scope size
            num_variables = max(circuit_scope) + 1
            state = torch.full(
                (num_samples, num_variables), 0, dtype=torch.float, device=self._circuit.device
            )
            evidence_vars = state.clone().to(torch.bool)
        else:
            x = x.to(self._circuit.device)
            state = x.clone()

            if state.size(0) != 1:
                raise ValueError("Only one tensor can be provided as evidence.")

            state = state.tile((num_samples, 1))
            evidence_vars = evidence_vars.tile((num_samples, 1))

        samples_p, state = self._circuit.backtrack(
            x=state,
            module_fn=functools.partial(
                SamplingQuery._layer_fn, num_samples=num_samples, evidence_vars=evidence_vars
            ),
        )

        return samples_p, state

    @staticmethod
    def _layer_fn(
        layer: TorchLayer, x: Tensor, *, num_samples: int, evidence_vars: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the layer in the usual feedforward way by randomly sampling.

        Args:
            layer (TorchLayer): The layer for the forward pass.
            x (Tensor): The input to the layer.
            evidence_vars (Tensor): A tensor that indicates which variables are set
                as evidence.

        Returns:
            tuple[Tensor, Tensor]: A tuple which contains the input sampled the layer
                and its layer output.
        """
        if isinstance(layer, TorchInputLayer):
            if layer.num_variables:
                # sample from input layer
                sample_idx, sampled_output = layer.sample(num_samples)
                is_evidence = evidence_vars[:, layer.scope_idx].permute(1, 0, 2)
                # select evidence state where specified
                idx = torch.where(is_evidence, x, sample_idx)
                # when evidence is used, select that layer otherwise marginalize
                # the variable
                output = torch.where(is_evidence, layer(x), layer.integrate())
            else:
                # layer has been marginalized
                output = layer(x)
                idx = torch.full_like(output, torch.nan)
        else:
            idx, output = layer.sample(x)

        return idx, output
