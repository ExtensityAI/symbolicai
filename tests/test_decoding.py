from collections.abc import Callable

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

from symai.decoding import (
    MISSING,
    ConstructorDecoder,
    DecodeError,
    Missing,
    PydanticDecoder,
    TextDecoder,
    TypeAdapterDecoder,
    decode_output,
)
from symai.runtime.models import (
    AssistantOutputMessage,
    FinishReason,
    LanguageModelOutput,
    LanguageModelResponse,
    ResponseMetadata,
    TextContent,
)

METADATA = ResponseMetadata(
    provider="openai",
    requested_model="test-model",
    status_code=200,
    request_id="request-1",
)


def response(*outputs: tuple[int, str]) -> LanguageModelResponse:
    return LanguageModelResponse(
        outputs=tuple(
            LanguageModelOutput(
                index=index,
                message=AssistantOutputMessage(content=(TextContent(text=text),)),
                finish_reason=FinishReason.STOP,
            )
            for index, text in outputs
        ),
        metadata=METADATA,
    )


def test_text_and_constructor_decoders_normalize_scalar_text() -> None:
    normalized = response((0, "  'answer'  "))

    assert decode_output(normalized, TextDecoder()) == "answer"
    assert decode_output(response((0, " 42 ")), ConstructorDecoder(int)) == 42
    assert normalized.metadata is METADATA


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("true", True),
        (" YES ", True),
        ("1", True),
        ("false", False),
        ("No", False),
        ("0", False),
    ],
)
def test_constructor_decoder_accepts_explicit_boolean_forms(
    text: str,
    expected: bool,
) -> None:
    assert decode_output(response((0, text)), ConstructorDecoder(bool)) is expected


def test_constructor_decoder_rejects_unknown_boolean_text() -> None:
    with pytest.raises(ValueError, match="boolean"):
        decode_output(response((0, "possibly")), ConstructorDecoder(bool))


def test_type_adapter_decoder_preserves_nested_parameterized_types() -> None:
    decoder = TypeAdapterDecoder(TypeAdapter(list[dict[str, tuple[int, ...]]]))

    result = decode_output(
        response((0, '[{"scores": [1, 2]}, {"scores": [3]}]')),
        decoder,
    )

    assert result == [{"scores": (1, 2)}, {"scores": (3,)}]


def test_pydantic_decoder_validates_a_model() -> None:
    class Answer(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")

        value: int

    result = decode_output(
        response((0, '{"value": 7}')),
        PydanticDecoder(Answer),
    )

    assert result == Answer(value=7)


def test_output_selection_uses_normalized_index_not_tuple_position() -> None:
    normalized = response((2, "third"), (0, "first"), (1, "second"))

    assert decode_output(normalized, TextDecoder(), output_index=1) == "second"
    with pytest.raises(IndexError, match="output index 4"):
        decode_output(
            normalized,
            TextDecoder(),
            output_index=4,
            default="fallback",
        )


def test_explicit_default_catches_only_decoder_failure_and_is_limited() -> None:
    assert isinstance(MISSING, Missing)
    assert decode_output(
        response((0, "not-a-list")),
        ConstructorDecoder(list),
        default=["fallback", "extra", "ignored"],
        limit=2,
    ) == ["fallback", "extra"]
    assert (
        decode_output(
            response((0, "not-an-integer")),
            ConstructorDecoder(int),
            default=None,
        )
        is None
    )
    with pytest.raises(DecodeError) as error:
        decode_output(response((0, "not-an-integer")), ConstructorDecoder(int))
    assert isinstance(error.value.__cause__, ValueError)


def test_explicit_decode_error_uses_default_and_propagates_without_one() -> None:
    class RejectingDecoder:
        def decode(self, _text: str, /) -> str:
            msg = "decoder rejected output"
            raise DecodeError(msg)

    assert (
        decode_output(
            response((0, "value")),
            RejectingDecoder(),
            default="fallback",
        )
        == "fallback"
    )
    with pytest.raises(DecodeError, match="decoder rejected output"):
        decode_output(response((0, "value")), RejectingDecoder())


def test_default_does_not_hide_unexpected_decoder_exception() -> None:
    class ExplodingDecoder:
        def decode(self, _text: str, /) -> str:
            msg = "decoder bug"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="decoder bug"):
        decode_output(
            response((0, "value")),
            ExplodingDecoder(),
            default="fallback",
        )


def test_default_does_not_catch_output_selection_or_limiting_errors() -> None:
    class RecordingDecoder:
        def __init__(self, decode: Callable[[str], object]) -> None:
            self._decode = decode
            self.calls: list[str] = []

        def decode(self, text: str, /) -> object:
            self.calls.append(text)
            return self._decode(text)

    decoder = RecordingDecoder(lambda text: text)
    with pytest.raises(IndexError):
        decode_output(response((0, "value")), decoder, output_index=3, default="fallback")
    assert decoder.calls == []

    with pytest.raises(ValueError, match="greater than zero"):
        decode_output(response((0, "value")), decoder, default="fallback", limit=0)


def test_collection_limiting_preserves_sequence_and_mapping_order() -> None:
    assert decode_output(
        response((0, "[1, 2, 3]")),
        TypeAdapterDecoder(TypeAdapter(list[int])),
        limit=2,
    ) == [1, 2]
    assert decode_output(
        response((0, "[1, 2, 3]")),
        TypeAdapterDecoder(TypeAdapter(tuple[int, ...])),
        limit=2,
    ) == (1, 2)
    assert decode_output(
        response((0, '{"first": 1, "second": 2, "third": 3}')),
        TypeAdapterDecoder(TypeAdapter(dict[str, int])),
        limit=2,
    ) == {"first": 1, "second": 2}


@pytest.mark.parametrize(
    ("decoder", "expected"),
    [
        (TypeAdapterDecoder(TypeAdapter(set[int])), {1, 2, 3}),
        (
            TypeAdapterDecoder(TypeAdapter(frozenset[int])),
            frozenset({1, 2, 3}),
        ),
    ],
)
def test_collection_limiting_leaves_unordered_collections_unchanged(
    decoder: TypeAdapterDecoder[set[int]] | TypeAdapterDecoder[frozenset[int]],
    expected: set[int] | frozenset[int],
) -> None:
    assert decode_output(response((0, "[1, 2, 3]")), decoder, limit=1) == expected
