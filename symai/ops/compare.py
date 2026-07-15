from __future__ import annotations

from symai.decoding import ConstructorDecoder
from symai.function import Function
from symai.ops.primitives import _execute_language
from symai.prompts import ContainsValue, FuzzyEquals, IsInstanceOf
from symai.runtime.runtime import Runtime
from symai.symbol import Symbol

__all__ = ("equals", "contains", "is_instance_of")

_EQUALS_EXAMPLES = tuple(FuzzyEquals().value)
_CONTAINS_EXAMPLES = tuple(ContainsValue().value)
_IS_INSTANCE_OF_EXAMPLES = tuple(IsInstanceOf().value)
_BOOLEAN_DECODER = ConstructorDecoder(bool)


def equals[LeftT, RightT](
    runtime: Runtime,
    left: Symbol[LeftT],
    right: Symbol[RightT],
    *,
    engine: str | None = None,
) -> Symbol[bool]:
    left_value = _symbol_value(left, "left")
    right_value = _symbol_value(right, "right")
    function = Function(
        "Make a fuzzy equality comparison. "
        "Are the following objects contextually the same?\n",
        examples=_EQUALS_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"{left_value!s} == {right_value!s} =>",),
        _BOOLEAN_DECODER,
        engine=engine,
    )


def contains[ContainerT, ElementT](
    runtime: Runtime,
    container: Symbol[ContainerT],
    element: Symbol[ElementT],
    *,
    engine: str | None = None,
) -> Symbol[bool]:
    container_value = _symbol_value(container, "container")
    element_value = _symbol_value(element, "element")
    function = Function(
        "Is the information in 'A' semantically contained in 'B'?\n",
        examples=_CONTAINS_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"{element_value!s} in {container_value!s} =>",),
        _BOOLEAN_DECODER,
        engine=engine,
    )


def is_instance_of[T](
    runtime: Runtime,
    source: Symbol[T],
    type_description: str,
    *,
    engine: str | None = None,
) -> Symbol[bool]:
    value = _symbol_value(source, "source")
    if not isinstance(type_description, str):
        msg = "type_description must be text"
        raise TypeError(msg)

    function = Function(
        "Is 'A' semantically an instance of the described type 'B'?\n",
        examples=_IS_INSTANCE_OF_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"{value!s} is instance of {type_description} =>",),
        _BOOLEAN_DECODER,
        engine=engine,
    )


def _symbol_value[T](symbol: Symbol[T], field: str) -> T:
    if not isinstance(symbol, Symbol):
        msg = f"{field} must be a Symbol"
        raise TypeError(msg)

    return symbol.value
