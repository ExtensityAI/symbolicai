from symai.decoding import Decoder, decode_output
from symai.function import Function
from symai.runtime.runtime import Runtime
from symai.symbol import Symbol


def _execute_language[T](
    runtime: Runtime,
    function: Function,
    values: tuple[object, ...],
    decoder: Decoder[T],
    *,
    engine: str | None,
) -> Symbol[T]:
    response = function(runtime, *values, engine=engine)
    return Symbol(decode_output(response, decoder))
