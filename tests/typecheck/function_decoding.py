from typing import assert_type

from pydantic import BaseModel, TypeAdapter

from symai.decoding import decode_bool, decode_output, decode_text, scalar_decoder
from symai.function import Function
from symai.runtime.models import LanguageModelResponse


class Answer(BaseModel):
    value: int


def prove_decoder_result_inference(response: LanguageModelResponse) -> None:
    assert_type(response.text, str)
    assert_type(decode_output(response, decode_text), str)
    assert_type(decode_output(response, decode_bool), bool)
    assert_type(decode_output(response, int), int)
    assert_type(decode_output(response, scalar_decoder(int)), int)
    assert_type(
        decode_output(response, TypeAdapter(list[dict[str, int]]).validate_json),
        list[dict[str, int]],
    )
    assert_type(decode_output(response, TypeAdapter(Answer).validate_json), Answer)


function = Function("Answer.")
assert_type(function, Function)
Function[int]("Answer.")  # pyright: ignore[reportInvalidTypeArguments]
