from __future__ import annotations

from symai.decoding import TextDecoder
from symai.function import Function
from symai.ops.primitives import _execute_language
from symai.runtime.runtime import Runtime
from symai.symbol import Symbol

__all__ = ("rank",)

_RANK_EXAMPLES = ("order: 'desc' measure: 'ASCII occurrence' list: ['b', 'a', 'z', 3, '_'] =>['_', 3, 'a', 'b', "
 "'z']",
 "order: 'desc' measure: 'Value' list: ['Action: l Value: -inf', 'Action: r Value: 0.76', 'Action: "
 "u Value: 0.76', 'Action: d Value: 0.00'] =>['Action: r Value: 0.76', 'Action: u Value: 0.76', "
 "'Action: d Value: 0.00', 'Action: l Value: -inf']",
 "order: 'asc' measure: 'Number' list: ['Number: -0.26', 'Number: -0.37', 'Number: 0.76', 'Number: "
 "-inf', 'Number: inf', 'Number: 0.37', 'Number: 1.0', 'Number: 100'] =>['Number: -inf', 'Number: "
 "-0.37', 'Number: -0.26', 'Number: 0.37', 'Number: 0.76', 'Number: 1.0', 'Number: 100', 'Number: "
 "inf']",
 "order: 'asc' measure: 'ASCII occurrence' list: ['b', 'a', 'z', 3, '_'] =>['z', 'b', 'a', 3, '_']",
 "order: 'desc' measure: 'length' list: [33, 'a', , 'help', 1234567890] =>['a', 33, 'help', "
 '1234567890]',
 "order: 'asc' measure: 'length' list: [33, 'a', , 'help', 1234567890] =>[1234567890, 'help', 'a', "
 '33]',
 "order: 'desc' measure: 'numeric size' list: [100, -1, 0, 1e-5, 1e-6] =>[100, 1e-5, 1e-6, 0, -1]",
 "order: 'asc' measure: 'numeric size' list: [100, -1, 0, 1e-5, 1e-6] =>[-1, 0, 1e-5, 1e-6, 100]",
 "order: 'desc' measure: 'fruits alphabetic' list: ['banana', 'orange', 'apple', 'pear'] "
 "=>['apple', 'banana', 'orange', 'pear']",
 "order: 'asc' measure: 'fruits alphabetic' list: ['banana', 'orange', 'horse', 'apple', 'pear'] "
 "=>['horse', 'pear', 'orange', 'banana', 'apple']",
 "order: 'desc' measure: 'HEX order in ASCII' list: [1, '1', 2, '2', 3, '3'] =>[1, 2, 3, '1', '2', "
 "'3']",
 "order: 'asc' measure: 'HEX order in ASCII' list: [1, '1', 2, '2', 3, '3'] =>['3', '2', '1', 3, "
 '2, 1]',
 "order: 'desc' measure: 'house building order' list: ['construct the roof', 'gather materials', "
 "'buy land', 'build the walls', 'dig the foundation'] =>['buy land', 'gather materials', 'dig the "
 "foundation', 'build the walls', 'construct the roof']")


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
