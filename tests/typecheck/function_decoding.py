from typing import assert_type

from pydantic import BaseModel

from symai.decoding import (
    ConstructorDecoder,
    TextDecoder,
    TypeAdapterDecoder,
    decode_output,
)
from symai.function import Function
from symai.runtime.models import LanguageModelResponse


class Answer(BaseModel):
    value: int


def prove_decoder_result_inference(response: LanguageModelResponse) -> None:
    assert_type(decode_output(response, TextDecoder()), str)
    assert_type(decode_output(response, ConstructorDecoder(int)), int)
    assert_type(
        decode_output(
            response,
            TypeAdapterDecoder(list[dict[str, int]]),
        ),
        list[dict[str, int]],
    )
    assert_type(decode_output(response, TypeAdapterDecoder(Answer)), Answer)


function = Function("Answer.")
assert_type(function, Function)
Function[int]("Answer.")  # pyright: ignore[reportInvalidTypeArguments]
