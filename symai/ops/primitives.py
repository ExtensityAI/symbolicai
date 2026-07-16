from symai.decoding import Decoder, decode_output
from symai.function import Function
from symai.runtime.runtime import LanguageModel
from symai.symbol import Symbol


def _symbol_value[T](symbol: Symbol[T], field: str) -> T:
    if not isinstance(symbol, Symbol):
        msg = f"{field} must be a Symbol"
        raise TypeError(msg)

    return symbol.value


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str):
        msg = f"{field} must be text"
        raise TypeError(msg)


def _execute_language[T](
    model: LanguageModel,
    function: Function,
    values: tuple[object, ...],
    decoder: Decoder[T],
) -> Symbol[T]:
    response = function(model, *values)
    return Symbol(decode_output(response, decoder))
