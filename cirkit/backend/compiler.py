from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar, cast

from cirkit.backend.registry import CompilerRegistry
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import Initializer
from cirkit.symbolic.layers import Layer
from cirkit.symbolic.parameters import ParameterNode
from cirkit.utils.algorithms import BiMap

SUPPORTED_BACKENDS = ["torch"]

CompiledCircuitT = TypeVar("CompiledCircuitT")

LayerCompilationSign = type[Layer]
ParameterCompilationSign = type[ParameterNode]
InitializerCompilationSign = type[Initializer]


class CompiledCircuitsMap(Generic[CompiledCircuitT]):
    def __init__(self) -> None:
        self._bimap = BiMap[Circuit, CompiledCircuitT]()

    def is_compiled(self, sc: Circuit) -> bool:
        return self._bimap.has_left(sc)

    def has_symbolic(self, cc: CompiledCircuitT) -> bool:
        return self._bimap.has_right(cc)

    def get_compiled_circuit(self, sc: Circuit) -> CompiledCircuitT:
        return self._bimap.get_left(sc)

    def get_symbolic_circuit(self, cc: CompiledCircuitT) -> Circuit:
        return self._bimap.get_right(cc)

    def register_compiled_circuit(self, sc: Circuit, cc: CompiledCircuitT) -> None:
        self._bimap.add(sc, cc)


class LayerCompilationFunc(Protocol):
    """The layer compilation function protocol."""

    def __call__(self, compiler: "AbstractCompiler[CompiledCircuitT]", sl: Layer) -> Any:
        """Compile a symbolic layer, given a compiler.

        Args:
            compiler: The compiler.
            sl: The symbolic layer.

        Returns:
            A representation of the compiled layer,
                which depends on the chosen compilation backend.
        """


class ParameterCompilationFunc(Protocol):
    """The parameter node compilation function protocol."""

    def __call__(self, compiler: "AbstractCompiler[CompiledCircuitT]", p: ParameterNode) -> Any:
        """Compile a symbolic parameter node, given a compiler.

        Args:
            compiler: The compiler.
            p: The symbolic parameter node.

        Returns:
            A representation of the compiled parameter node,
                which depends on the chosen compilation backend.
        """


class InitializerCompilationFunc(Protocol):
    """The initialization method compilation function protocol."""

    def __call__(self, compiler: "AbstractCompiler[CompiledCircuitT]", init: Initializer) -> Any:
        """Compile a symbolic initializer, given a compiler.

        Args:
            compiler: The compiler.
            init: The symbolic initializer.

        Returns:
            A representation of the compiled initializer,
                which depends on the chosen compilation backend.
        """


class CompilationRuleNotFound(Exception):
    """An exception that is raised when a compilation rule is not found."""

    def __init__(self, msg: str):
        """Initializes a compilation rule not found exception.

        Args:
            msg: The message of the exception.
        """
        super().__init__(msg)


class CompilerLayerRegistry(CompilerRegistry[LayerCompilationSign, LayerCompilationFunc]):
    @classmethod
    def _validate_rule_function(cls, func: LayerCompilationFunc) -> bool:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return issubclass(ann[args[-1]], Layer)

    @classmethod
    def _retrieve_signature(cls, func: LayerCompilationFunc) -> LayerCompilationSign:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return cast(LayerCompilationSign, ann[args[-1]])


class CompilerParameterRegistry(
    CompilerRegistry[ParameterCompilationSign, ParameterCompilationFunc]
):
    @classmethod
    def _validate_rule_function(cls, func: ParameterCompilationFunc) -> bool:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return issubclass(ann[args[-1]], ParameterNode)

    @classmethod
    def _retrieve_signature(cls, func: ParameterCompilationFunc) -> ParameterCompilationSign:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return cast(ParameterCompilationSign, ann[args[-1]])


class CompilerInitializerRegistry(
    CompilerRegistry[
        InitializerCompilationSign,
        InitializerCompilationFunc,
    ]
):
    @classmethod
    def _validate_rule_function(cls, func: InitializerCompilationFunc) -> bool:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return issubclass(ann[args[-1]], Initializer)

    @classmethod
    def _retrieve_signature(cls, func: InitializerCompilationFunc) -> InitializerCompilationSign:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return cast(InitializerCompilationSign, ann[args[-1]])


class AbstractCompiler(ABC, Generic[CompiledCircuitT]):
    def __init__(
        self,
        layers_registry: CompilerLayerRegistry,
        parameters_registry: CompilerParameterRegistry,
        initializers_registry: CompilerInitializerRegistry,
        **flags: Any,
    ) -> None:
        self._layers_registry = layers_registry
        self._parameters_registry = parameters_registry
        self._initializers_registry = initializers_registry
        self._flags = flags
        self._compiled_circuits = CompiledCircuitsMap[CompiledCircuitT]()

    def is_compiled(self, sc: Circuit) -> bool:
        return self._compiled_circuits.is_compiled(sc)

    def has_symbolic(self, cc: CompiledCircuitT) -> bool:
        return self._compiled_circuits.has_symbolic(cc)

    def get_compiled_circuit(self, sc: Circuit) -> CompiledCircuitT:
        return self._compiled_circuits.get_compiled_circuit(sc)

    def get_symbolic_circuit(self, cc: CompiledCircuitT) -> Circuit:
        return self._compiled_circuits.get_symbolic_circuit(cc)

    def register_compiled_circuit(self, sc: Circuit, cc: CompiledCircuitT) -> None:
        self._compiled_circuits.register_compiled_circuit(sc, cc)

    def add_layer_rule(self, func: LayerCompilationFunc) -> None:
        self._layers_registry.add_rule(func)

    def add_parameter_rule(self, func: ParameterCompilationFunc) -> None:
        self._parameters_registry.add_rule(func)

    def add_initializer_rule(self, func: InitializerCompilationFunc) -> None:
        self._initializers_registry.add_rule(func)

    def retrieve_layer_rule(self, signature: LayerCompilationSign) -> LayerCompilationFunc:
        return self._layers_registry.retrieve_rule(signature)

    def retrieve_parameter_rule(
        self, signature: ParameterCompilationSign
    ) -> ParameterCompilationFunc:
        return self._parameters_registry.retrieve_rule(signature)

    def retrieve_initializer_rule(
        self, signature: InitializerCompilationSign
    ) -> InitializerCompilationFunc:
        return self._initializers_registry.retrieve_rule(signature)

    def compile(self, sc: Circuit) -> CompiledCircuitT:
        if self.is_compiled(sc):
            return self.get_compiled_circuit(sc)
        return self.compile_pipeline(sc)

    @abstractmethod
    def compile_pipeline(self, sc: Circuit) -> CompiledCircuitT: ...
