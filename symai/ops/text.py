from __future__ import annotations

from symai.decoding import TextDecoder
from symai.function import Function
from symai.ops.primitives import _execute_language
from symai.prompts import (
    CombineText,
    ExtractPattern,
    Format,
    IncludeText,
    MapExpression,
    Modify,
    ReplaceText,
)
from symai.runtime.runtime import Runtime
from symai.symbol import Symbol

__all__ = (
    "summarize",
    "translate",
    "modify",
    "filter",
    "map",
    "convert",
    "style",
    "template",
    "replace",
    "include",
    "combine",
    "extract",
)

_MODIFY_EXAMPLES = tuple(Modify().value)
_MAP_EXAMPLES = tuple(MapExpression().value)
_FORMAT_EXAMPLES = tuple(Format().value)
_REPLACE_EXAMPLES = tuple(ReplaceText().value)
_INCLUDE_EXAMPLES = tuple(IncludeText().value)
_COMBINE_EXAMPLES = tuple(CombineText().value)
_EXTRACT_EXAMPLES = tuple(ExtractPattern().value)


def summarize[T](
    runtime: Runtime,
    source: Symbol[T],
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    function = Function("Summarize the content of the following text:\n")
    return _execute_language(
        runtime,
        function,
        (f"Text: {value!s}\n",),
        TextDecoder(),
        engine=engine,
    )


def translate[T](
    runtime: Runtime,
    source: Symbol[T],
    language: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(language, "language")
    function = Function(
        f"Your task is to translate and **only** translate the text into {language}:\n"
    )
    return _execute_language(
        runtime,
        function,
        (str(value),),
        TextDecoder(),
        engine=engine,
    )


def modify[T](
    runtime: Runtime,
    source: Symbol[T],
    changes: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(changes, "changes")
    function = Function(
        "Modify the text to match the criteria:\n",
        examples=_MODIFY_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"text '{value!s}' modify '{changes}'=>",),
        TextDecoder(),
        engine=engine,
    )


def filter[T](
    runtime: Runtime,
    source: Symbol[T],
    criteria: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(criteria, "criteria")
    function = Function(
        "Filter the text to retain only information matching the criteria. "
        "Leave matching sentences unchanged:\n"
    )
    return _execute_language(
        runtime,
        function,
        (f"text '{value!s}' criteria '{criteria}' =>",),
        TextDecoder(),
        engine=engine,
    )


def map[T](
    runtime: Runtime,
    source: Symbol[T],
    instruction: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(instruction, "instruction")
    function = Function(
        "Transform each element in the input based on the instruction. "
        "Preserve container type and elements that don't match the instruction:\n",
        examples=_MAP_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"text '{value!s}' {instruction} =>",),
        TextDecoder(),
        engine=engine,
    )


def convert[T](
    runtime: Runtime,
    source: Symbol[T],
    format: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(format, "format")
    function = Function(
        f"Translate the following text into {format} format.\n",
        examples=_FORMAT_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"text {value!s} format '{format}' =>",),
        TextDecoder(),
        engine=engine,
    )


def style[T](
    runtime: Runtime,
    source: Symbol[T],
    description: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(description, "description")
    function = Function(
        "Style the data based on best practices and the requested description. "
        "Do not remove or invent content.\n"
    )
    return _execute_language(
        runtime,
        function,
        (f"[FORMAT]: {description}\n[DATA]:\n{value!s}\n",),
        TextDecoder(),
        engine=engine,
    )


def template[T](
    source: Symbol[T],
    template: str,
    *,
    placeholder: str = "{{placeholder}}",
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(template, "template")
    _require_text(placeholder, "placeholder")
    if not placeholder:
        msg = "placeholder must not be empty"
        raise ValueError(msg)

    return Symbol(template.replace(placeholder, str(value)))


def replace[T](
    runtime: Runtime,
    source: Symbol[T],
    old: str,
    new: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(old, "old")
    _require_text(new, "new")
    function = Function(
        "Replace text parts by string pattern.\n",
        examples=_REPLACE_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"text '{value!s}' replace '{old}' with '{new}'=>",),
        TextDecoder(),
        engine=engine,
    )


def include[T](
    runtime: Runtime,
    source: Symbol[T],
    information: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(information, "information")
    function = Function(
        "Include information based on description.\n",
        examples=_INCLUDE_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"text '{value!s}' include '{information}' =>",),
        TextDecoder(),
        engine=engine,
    )


def combine[LeftT, RightT](
    runtime: Runtime,
    left: Symbol[LeftT],
    right: Symbol[RightT],
    *,
    engine: str | None = None,
) -> Symbol[str]:
    left_value = _symbol_value(left, "left")
    right_value = _symbol_value(right, "right")
    function = Function(
        "Add the two data types in a logical way:\n",
        examples=_COMBINE_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"{left_value!s} + {right_value!s} =>",),
        TextDecoder(),
        engine=engine,
    )


def extract[T](
    runtime: Runtime,
    source: Symbol[T],
    pattern: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(pattern, "pattern")
    function = Function(
        "Extract a pattern from text:\n",
        examples=_EXTRACT_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"from '{value!s}' extract '{pattern}' =>",),
        TextDecoder(),
        engine=engine,
    )


def _symbol_value[T](symbol: Symbol[T], field: str) -> T:
    if not isinstance(symbol, Symbol):
        msg = f"{field} must be a Symbol"
        raise TypeError(msg)

    return symbol.value


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str):
        msg = f"{field} must be text"
        raise TypeError(msg)
