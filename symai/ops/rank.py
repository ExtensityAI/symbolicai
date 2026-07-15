from __future__ import annotations

from symai.decoding import TextDecoder
from symai.function import Function
from symai.ops.primitives import _execute_language
from symai.prompts import RankList
from symai.runtime.runtime import Runtime
from symai.symbol import Symbol

__all__ = ("rank",)

_RANK_EXAMPLES = tuple(RankList().value)


def rank[T](
    runtime: Runtime,
    source: Symbol[T],
    measure: str,
    *,
    engine: str | None = None,
) -> Symbol[str]:
    if not isinstance(source, Symbol):
        msg = "source must be a Symbol"
        raise TypeError(msg)
    if not isinstance(measure, str):
        msg = "measure must be text"
        raise TypeError(msg)

    function = Function(
        "Rank the objects from highest to lowest by the requested measure:\n",
        examples=_RANK_EXAMPLES,
    )
    return _execute_language(
        runtime,
        function,
        (f"measure: '{measure}' list: {source.value!s} =>",),
        TextDecoder(),
        engine=engine,
    )
