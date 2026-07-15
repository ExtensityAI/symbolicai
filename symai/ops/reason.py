from __future__ import annotations

from symai.decoding import TextDecoder
from symai.function import Function
from symai.ops.primitives import _execute_language
from symai.prompts import LogicExpression, SimpleSymbolicExpression
from symai.runtime.runtime import Runtime
from symai.symbol import Symbol

__all__ = ("query", "interpret", "logic")

_INTERPRET_EXAMPLES = tuple(SimpleSymbolicExpression().value)
_LOGIC_EXAMPLES = tuple(LogicExpression().value)


def query[T](
    runtime: Runtime,
    source: Symbol[T],
    question: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(question, "question")
    function = Function("Answer the question using only the provided data:\n")
    return _execute_language(
        runtime,
        function,
        (f"Data:\n{value!s}\nQuestion: {question}\nAnswer:",),
        TextDecoder(),
        engine=engine,
    )


def interpret[T](
    runtime: Runtime,
    source: Symbol[T],
    *,
    engine: str | None = None,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    function = Function(
        "Evaluate the symbolic expression and return only the result:\n",
        examples=_INTERPRET_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"{value!s} =>",),
        TextDecoder(),
        engine=engine,
    )


def logic[LeftT, RightT](
    runtime: Runtime,
    left: Symbol[LeftT],
    operator: str,
    right: Symbol[RightT],
    *,
    engine: str | None = None,
) -> Symbol[str]:
    left_value = _symbol_value(left, "left")
    right_value = _symbol_value(right, "right")
    _require_text(operator, "operator")
    function = Function(
        "Evaluate the logic expression:\n",
        examples=_LOGIC_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"expr :{left_value!s}: {operator} :{right_value!s}: =>",),
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
