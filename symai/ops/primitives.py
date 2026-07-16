from symai.decoding import Decoder, decode_output
from symai.function import Function
from symai.runtime.runtime import LanguageModel
from symai.symbol import Symbol


def _execute_language[T](
    model: LanguageModel,
    function: Function,
    values: tuple[object, ...],
    decoder: Decoder[T],
) -> Symbol[T]:
    response = function(model, *values)
    return Symbol(decode_output(response, decoder))
