import copy
import html
import json
from collections.abc import Iterator
from json import JSONEncoder
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
from box import Box

from symai.operations import language_request, parse_stripped_output
from symai.ops.primitives import (
    CastingPrimitives,
    ComparisonPrimitives,
    DataHandlingPrimitives,
    EmbeddingPrimitives,
    ExpressionHandlingPrimitives,
    IterationPrimitives,
    OperatorPrimitives,
    PatternMatchingPrimitives,
    PersistencePrimitives,
    QueryHandlingPrimitives,
    StringHelperPrimitives,
    TemplateStylingPrimitives,
    ValueHandlingPrimitives,
)
from symai.runtime.runtime import current_runtime

T = TypeVar("T")


class SymbolEncoder(JSONEncoder):
    def default(self, o):
        """
        Encode a Symbol instance into its dictionary representation.

        Args:
            sym (Symbol): The Symbol instance to encode.

        Returns:
            dict: The dictionary representation of the Symbol instance.
        """
        if isinstance(o, Symbol):
            return o.__getstate__()
        return JSONEncoder.default(self, o)


class Linker:
    """Ordered results produced while evaluating an expression graph."""

    __slots__ = ("results",)

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}

    def keys(self) -> list[str]:
        return list(self.results)

    def values(self) -> list[Any]:
        return list(self.results.values())

    def find(self, name: str, single: bool = True, strict: bool = False) -> Any:
        def matches(key: str) -> bool:
            if strict:
                return str(name) == str(key)
            return str(name).lower() in str(key).lower()

        results = [value for key, value in self.results.items() if matches(key)]
        if single and len(results) != 1:
            msg = f"Found {len(results)} results for name {name}. Expected 1."
            raise ValueError(msg)
        if not results:
            return None
        if len(results) == 1:
            return results[0]
        return results


class PropertyReservedError(AttributeError):
    def __init__(self, property_name):
        self.property_name = property_name
        super().__init__(
            f"Cannot set reserved property '{property_name}'. This property is defined in the 'Symbol' base class and cannot be assigned."
        )


class SymbolMeta(type(OperatorPrimitives)):
    """Run graph linking after concrete Symbol initialization."""

    def __call__(cls, *args: object, **kwargs: object) -> Any:
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj


class Symbol(
    OperatorPrimitives,
    IterationPrimitives,
    ValueHandlingPrimitives,
    StringHelperPrimitives,
    CastingPrimitives,
    ComparisonPrimitives,
    ExpressionHandlingPrimitives,
    DataHandlingPrimitives,
    PatternMatchingPrimitives,
    QueryHandlingPrimitives,
    TemplateStylingPrimitives,
    EmbeddingPrimitives,
    PersistencePrimitives,
    Generic[T],
    metaclass=SymbolMeta,
):
    _dynamic_context: ClassVar[dict[str, list[str]]] = {}
    _RESERVED_PROPERTIES: ClassVar[set[str]] = {
        "graph",
        "linker",
        "parent",
        "children",
        "value",
        "root",
        "nodes",
        "edges",
        "global_context",
        "static_context",
        "dynamic_context",
        "shape",
    }

    def __init__(
        self,
        *value: object,
        static_context: str | None = None,
        dynamic_context: str | None = None,
        semantic: bool | None = None,
    ) -> None:
        """Create a value with optional context and explicit-runtime fallback mode."""

        nested = value[0] if len(value) == 1 and isinstance(value[0], Symbol) else None
        if semantic is not None and not isinstance(semantic, bool):
            msg = "semantic must be a boolean or None"
            raise TypeError(msg)

        self._value: object = None
        self._parent: Symbol | None = None
        self._children: list[Symbol] = []
        self._static_context = (
            nested.static_context
            if nested is not None and static_context is None
            else static_context
        )
        self._dynamic_context_value: str | dict[str, list[str]] = (
            nested._dynamic_context_value
            if nested is not None and dynamic_context is None
            else dynamic_context or Symbol._dynamic_context
        )
        self._semantic = (
            nested._semantic if nested is not None and semantic is None else bool(semantic)
        )
        self._embedding: np.ndarray | None = None
        self._detached = False
        self._root_linker: Linker | None = None
        self._value = self._unwrap_symbols_args(*value)
        self._construct_dependency_graph(*value)

    def __post_init__(self, *args: object, **kwargs: object) -> None:
        def link_value(name: str, value: object) -> None:
            if isinstance(value, Symbol) and not name.startswith("_") and value is not self:
                value._parent = self
                self._children.append(value)
            elif isinstance(value, (list, tuple)) and not name.startswith("_"):
                for item in value:
                    link_value(name, item)

        for name, value in self.__dict__.items():
            link_value(name, value)

    def _unwrap_symbols_args(self, *args: object) -> Any:
        if not args:
            return None
        if len(args) == 1:
            return self._unwrap_single_symbol_arg(args[0])
        return [self._unwrap_symbols_args(arg) if isinstance(arg, Symbol) else arg for arg in args]

    def _unwrap_single_symbol_arg(self, value: object) -> Any:
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Symbol):
            return value.value

        from box import Box, BoxList  # noqa: PLC0415

        if isinstance(value, (Box, BoxList)):
            return value
        if isinstance(value, list):
            return [self._unwrap_symbols_args(item) for item in value]
        if isinstance(value, dict):
            return {
                self._unwrap_symbols_args(key): self._unwrap_symbols_args(item)
                for key, item in value.items()
            }
        if isinstance(value, set):
            return {self._unwrap_symbols_args(item) for item in value}
        if isinstance(value, tuple):
            return tuple(self._unwrap_symbols_args(item) for item in value)
        return value

    def _construct_dependency_graph(self, *values: object) -> None:
        for value in values:
            if isinstance(value, Symbol) and value is not self:
                value._parent = self
                self._children.append(value)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set the attribute of the Symbol's value with the specified name.

        Args:
            name (str): The name of the attribute to set for the Symbol's value.
            value (Any): The value of the attribute to set for the Symbol's value.

        Raises:
            PropertyReservedError: If the property is reserved and cannot be set.
        """
        if name in self._RESERVED_PROPERTIES:
            raise PropertyReservedError(name)
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """
        Get the attribute of the Symbol's value with the specified name or the attribute of the Symbol value with the specified name.

        Args:
            name (str): The name of the attribute to get from the Symbol's value.

        Returns:
            Any: The attribute of the Symbol's value with the specified name.
        """
        try:
            # try to get attribute from current instance
            if name in self.__dict__:
                return self.__dict__[name]
            value = self.value if self.value is not None else None
            if isinstance(value, Exception):
                raise value
            msg = f"<class '{self.__class__.__name__}'> or nested value of {type(value)!s} have no attribute '{name}'"
            raise AttributeError(msg)
        except AttributeError as ex:
            # if has attribute and is public function
            if hasattr(self.value, name) and not name.startswith("_"):
                return getattr(self.value, name)
            raise ex

    def __array__(self, dtype=None):
        """
        Get the numpy array representation of the Symbol's value.

        Returns:
            np.ndarray: The numpy array representation of the Symbol's value.
        """
        return self.embedding.astype(dtype, copy=False)

    def __buffer__(self, flags=0):
        """
        Get the buffer of the Symbol's value.

        Args:
            flags (int, optional): The flags for the buffer. Defaults to 0.

        Returns:
            memoryview: The buffer of the Symbol's value.
        """
        return memoryview(self.embedding)

    @staticmethod
    def symbols(*values) -> list["Symbol"]:
        """
        Create a list of Symbol instances from a list of values.

        Args:
            values (List[Any]): The list of values to create Symbol instances from.

        Returns:
            List[Symbol]: The list of Symbol instances.
        """
        return [Symbol(value) for value in values]

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the symbol for serialization.

        Returns:
            dict: The state of the symbol.
        """
        state = vars(self).copy()
        state.pop("_embedding", None)
        state.pop("_parent", None)
        state.pop("_children", None)
        state.pop("_root_linker", None)
        state.pop("_detached", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)
        self._embedding = None
        self._parent = None
        self._children = []
        self._root_linker = None
        self._detached = False

    def json(self) -> dict[str, Any]:
        """
        Get the json-serializable dictionary representation of the Symbol instance.

        Returns:
            dict: The json-serializable dictionary representation of the Symbol instance.
        """
        return self.__getstate__()

    def serialize(self):
        """
        Encode an Symbol instance into its dictionary representation.

        Args:
            obj (Symbol): The Expression instance to encode.

        Returns:
            dict: The dictionary representation of the Symbol instance.
        """
        return json.dumps(self, cls=SymbolEncoder)

    def _to_symbol(self, value: Any, *, semantic: bool | None = None) -> "Symbol":
        if isinstance(value, Symbol):
            return value
        return Symbol(
            value,
            static_context=self.static_context,
            dynamic_context=self.dynamic_context,
            semantic=self._semantic if semantic is None else semantic,
        )

    def _to_type(self, value: Any, *, semantic: bool | None = None) -> "Symbol":
        symbol_type = type(self)
        if isinstance(value, symbol_type):
            return value
        return symbol_type(
            value,
            static_context=self.static_context,
            dynamic_context=self.dynamic_context,
            semantic=self._semantic if semantic is None else semantic,
        )

    @property
    def _symbol_type(self) -> type["Symbol"]:
        return Symbol

    def __hash__(self) -> int:
        """
        Get the hash value of the symbol.

        Returns:
            int: The hash value of the symbol.
        """
        return str(self.value).__hash__()

    @property
    def value(self) -> Any:
        """
        Get the value of the symbol.

        Returns:
            Any: The value of the symbol.
        """
        return self._value

    @property
    def global_context(self) -> tuple[str, str]:
        """Return stable and runtime context as an ordered pair."""

        return self.static_context, self.dynamic_context

    @property
    def static_context(self) -> str:
        """
        Get the static context of the symbol which is defined by the user when creating a symbol subclass.

        Returns:
            str: The static context of the symbol.
        """
        return f"{self._static_context}" if self._static_context else ""

    @static_context.setter
    def static_context(self, value: str):
        """
        Set the static context of the symbol which is defined by the user when creating a symbol subclass.
        """
        self._static_context = value

    @property
    def dynamic_context(self) -> str:
        """Return context supplied for the current explicit runtime request."""

        if isinstance(self._dynamic_context_value, str):
            return self._dynamic_context_value
        type_key = str(type(self))
        if type_key not in Symbol._dynamic_context:
            Symbol._dynamic_context[type_key] = []
            return ""
        values = Symbol._dynamic_context[type_key]
        if not values:
            return ""
        text = "\n".join(
            str(value.value) if isinstance(value, Symbol) else str(value) for value in values
        )
        return f"\n{text}" if text else ""

    @property
    def root(self) -> "Symbol":
        """
        Get the root of the symbol.

        Returns:
            Symbol: The root of the symbol.
        """
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    @property
    def nodes(self) -> list["Symbol"]:
        nodes: list[Symbol] = []

        def collect(node: Symbol) -> None:
            nodes.append(node)
            for child in node.children:
                collect(child)

        collect(self)
        return nodes

    @property
    def edges(self) -> list[tuple["Symbol", "Symbol"]]:
        edges: list[tuple[Symbol, Symbol]] = []

        def collect(node: Symbol) -> None:
            for child in node.children:
                edges.append((node, child))
                collect(child)

        collect(self)
        return edges

    @property
    def graph(self) -> tuple[list["Symbol"], list[tuple["Symbol", "Symbol"]]]:
        return self.nodes, self.edges

    @property
    def linker(self) -> Linker | None:
        """Return ordered expression results linked at the graph root."""

        return self.root._root_linker

    @property
    def parent(self) -> "Symbol | None":
        return self._parent

    @property
    def children(self) -> list["Symbol"]:
        """
        Get the children of the symbol.

        Returns:
            List[Symbol]: The children of the symbol.
        """
        return self._children

    def _root_link(self, symbol: Any, **_options: object) -> None:
        root = self.root
        if self is root or self._detached:
            return
        if root._root_linker is None:
            root._root_linker = Linker()

        previous = next(reversed(root._root_linker.results.values()), None)
        result = Symbol(symbol)
        if previous is not None and previous is not result.root:
            previous.children.append(result.root)
            result.root._parent = previous
        root._root_linker.results[self.__repr__()] = result

    def adapt(self, context: str, types: list[type] | None = None) -> None:
        """
        Update the dynamic context with a given runtime context.

        Args:
            context (str): The context to be added to the dynamic context.
            type (Type): The type used to update the dynamic context

        """
        if types is None:
            types = []
        if not isinstance(types, list):
            types = [types]
        if len(types) == 0:
            types = [type(self)]

        for type_ in types:
            type_key = str(type_)
            if type_key not in Symbol._dynamic_context:
                Symbol._dynamic_context[type_key] = []

            Symbol._dynamic_context[type_key].append(str(context))

    def clear(self, types: list[type] | None = None) -> None:
        """
        Clear the dynamic context associated with this symbol type.
        """
        if types is None:
            types = []
        if not isinstance(types, list):
            types = [types]
        if len(types) == 0:
            types = [type(self)]

        for type_ in types:
            type_key = str(type_)
            if type_key not in Symbol._dynamic_context:
                Symbol._dynamic_context[type_key] = []
                return

            Symbol._dynamic_context[type_key].clear()

    def __len__(self) -> int:
        return len(self.value)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.value.shape)

    def __str__(self) -> str:
        """
        Get the string representation of the Symbol's value.

        Returns:
            str: The string representation of the Symbol's value.
        """
        if self.value is None:
            return ""
        if isinstance(self.value, (list, np.ndarray, tuple)):
            return str([str(v) for v in self.value])
        if isinstance(self.value, dict):
            return str({k: str(v) for k, v in self.value.items()})
        if isinstance(self.value, set):
            return str({str(v) for v in self.value})
        return str(self.value)

    def __repr__(self, simplified: bool = False) -> str:
        """
        Get the representation of the Symbol object as a string.

        Returns:
            str: The representation of the Symbol object.
        """
        # class with full path
        class_ = self.__class__.__module__ + "." + self.__class__.__name__
        hex_ = hex(id(self))
        val = str(self.value)
        # only show first n characters of value and then add '...' and the last x characters
        if len(val) > 50:
            val = val[:25] + " ... " + val[-20:]
        return (
            f"<class {class_} at {hex_}>(value={val})" if not simplified else f"{class_} at {hex_}"
        )

    def _repr_html_(self) -> str:
        """
        Get the HTML representation of the Symbol's value.

        Returns:
            str: The HTML representation of the Symbol's value.
        """
        return html.escape(self.__repr__())

    def __iter__(self) -> Iterator:
        """
        Get an iterator for the Symbol's value.
        If the Symbol's value is a list, tuple, or numpy.ndarray, iterate over the elements. Otherwise, create a new list with a single item and iterate over the list.

        Returns:
            Iterator: An iterator for the Symbol's value.
        """
        if isinstance(self.value, (list, tuple, np.ndarray)):
            return iter(self.value)

        return iter((self.value,))

    def __reversed__(self) -> Iterator:
        """
        Get a reversed iterator for the Symbol's value.

        Returns:
            Iterator: A reversed iterator for the Symbol's value.
        """
        return reversed(list(self.__iter__()))

    def __next__(self) -> Any:
        """
        Get the next item in the iterable value of the Symbol.

        Returns:
            Symbol: The next item in the iterable value of the Symbol.

        Raises:
            StopIteration: If the iterable value reaches its end.
        """
        return next(self.__iter__())


class ExpressionEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Expression):
            return o.__getstate__()
        return JSONEncoder.default(self, o)


class Expression(Symbol):
    def __init__(
        self,
        value: object = None,
        *values: object,
        static_context: str | None = None,
        dynamic_context: str | None = None,
        semantic: bool | None = None,
    ) -> None:
        """Create a lazily evaluated value with explicit runtime context."""

        super().__init__(
            value,
            *values,
            static_context=static_context,
            dynamic_context=dynamic_context,
            semantic=semantic,
        )
        self._sym_return_type = type(self)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Evaluate the expression using the forward method and assign the result to the value attribute.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the forward method.
        """
        # evaluate the expression
        res = self.forward(*args, **kwargs)
        # store the result in the root node and link it to the previous result
        self._root_link(res, **kwargs)
        return res

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__().copy()
        state.pop("_sym_return_type", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._sym_return_type = type(self)

    def __json__(self):
        """
        Get the json-serializable dictionary representation of the Expression instance.

        Returns:
            dict: The json-serializable dictionary representation of the Expression instance.
        """
        return self.__getstate__()

    def serialize(self):
        """
        Encode an Expression instance into its dictionary representation.

        Args:
            obj (Expression): The Expression instance to encode.

        Returns:
            dict: The dictionary representation of the Expression instance.
        """
        return json.dumps(self, cls=ExpressionEncoder)

    @property
    def sym_return_type(self) -> type:
        """
        Returns the casting type of this expression.

        Returns:
            Type: The casting type of this expression. Defaults to the current Expression-type.
        """
        return self._sym_return_type

    @sym_return_type.setter
    def sym_return_type(self, symbol_type: type) -> None:
        self._sym_return_type = symbol_type

    def forward(self, *args: object, **kwargs: object) -> Symbol:
        """Evaluate this expression through its concrete implementation."""

        raise NotImplementedError

    def copy(self) -> Any:
        """
        Returns a deep copy of the own object.

        Returns:
            Any: A deep copy of the own object.
        """
        return copy.deepcopy(self)

    @staticmethod
    def prompt(
        message: str,
        *,
        output_index: int = 0,
        return_metadata: bool = False,
        return_type: type = str,
        default: object = None,
        limit: int | None = 1,
        **options: object,
    ) -> Any:
        """Execute one raw text request through the active runtime."""

        forbidden = {"engine", "model", "provider"}.intersection(options)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            msg = (
                "Provider/model selection belongs to runtime configuration, "
                f"not per-call kwargs: {names}"
            )
            raise TypeError(msg)
        if options:
            names = ", ".join(sorted(options))
            msg = f"Unsupported execution options: {names}"
            raise TypeError(msg)

        response = current_runtime().execute(language_request("", message))
        value = parse_stripped_output(
            response,
            return_type,
            index=output_index,
            default=default,
            limit=limit,
        )
        result = Expression(value)
        if return_metadata:
            return result, response.metadata
        return result


class Result(Expression):
    def __init__(
        self,
        value: object = None,
        *,
        static_context: str | None = None,
        dynamic_context: str | None = None,
        semantic: bool | None = None,
    ) -> None:
        """Create a result value with convenient structured access."""

        super().__init__(
            value,
            static_context=static_context,
            dynamic_context=dynamic_context,
            semantic=semantic,
        )
        self._sym_return_type = type(self)
        try:
            self.raw = Box(value)
        except Exception:
            self.raw = value

    @property
    def value(self) -> Any:
        """
        Get the value of the symbol.

        Returns:
            Any: The value of the symbol.
        """
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        """
        Set the value of the Result object.

        Args:
            value (Any): The value to set the Result object to.
        """
        self._value = value
