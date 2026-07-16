import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

from symai.decoding import (
    MISSING,
    DecodeError,
    Missing,
    decode_bool,
    decode_output,
    decode_text,
    scalar_decoder,
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


def test_response_text_returns_the_first_output_verbatim() -> None:
    normalized = response((0, "  raw text  "))

    assert normalized.text == "  raw text  "
    assert normalized.output_text() == "  raw text  "


def test_response_text_selects_by_index_not_tuple_position() -> None:
    normalized = response((2, "third"), (0, "first"), (1, "second"))

    assert normalized.text == "first"
    assert normalized.output_text(1) == "second"
    with pytest.raises(IndexError, match="output index 4"):
        normalized.output_text(4)


def test_decode_text_strips_whitespace_without_rewriting_quotes() -> None:
    normalized = response((0, "  'answer'  "))

    assert decode_output(normalized, decode_text) == "'answer'"
    assert normalized.metadata is METADATA


def test_decode_text_preserves_leading_apostrophe() -> None:
    assert decode_output(response((0, "  'Twas the night'  ")), decode_text) == ("'Twas the night'")


def test_scalar_decoder_normalizes_scalar_text() -> None:
    assert decode_output(response((0, "  '42'  ")), scalar_decoder(int)) == 42


def test_bare_constructor_decodes_clean_output_without_a_wrapper() -> None:
    assert decode_output(response((0, " 42 ")), int) == 42
    assert decode_output(response((0, "3.5")), float) == 3.5


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("true", True),
        (" YES ", True),
        ("1", True),
        ("false", False),
        ("No", False),
        ("'0'", False),
    ],
)
def test_decode_bool_accepts_explicit_boolean_forms(text: str, expected: bool) -> None:
    assert decode_output(response((0, text)), decode_bool) is expected
    assert decode_output(response((0, text)), scalar_decoder(bool)) is expected


def test_decode_bool_rejects_unknown_boolean_text() -> None:
    with pytest.raises(ValueError, match="boolean"):
        decode_output(response((0, "possibly")), decode_bool)


def test_scalar_decoder_rejects_container_types() -> None:
    with pytest.raises(TypeError, match="TypeAdapter"):
        scalar_decoder(list)


def test_type_adapter_preserves_nested_parameterized_types() -> None:
    decoder = TypeAdapter(list[dict[str, tuple[int, ...]]]).validate_json

    result = decode_output(response((0, '[{"scores": [1, 2]}, {"scores": [3]}]')), decoder)

    assert result == [{"scores": (1, 2)}, {"scores": (3,)}]


def test_type_adapter_validates_a_pydantic_model() -> None:
    class Answer(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")

        value: int

    result = decode_output(response((0, '{"value": 7}')), TypeAdapter(Answer).validate_json)

    assert result == Answer(value=7)


def test_type_adapter_does_not_strip_single_quotes_from_json() -> None:
    decoder = TypeAdapter(dict[str, int]).validate_json

    with pytest.raises(DecodeError):
        decode_output(response((0, """'{"value": 7}'""")), decoder)


def test_output_selection_uses_normalized_index_not_tuple_position() -> None:
    normalized = response((2, "third"), (0, "first"), (1, "second"))

    assert decode_output(normalized, decode_text, output_index=1) == "second"
    with pytest.raises(IndexError, match="output index 4"):
        decode_output(normalized, decode_text, output_index=4, default="fallback")


def test_explicit_default_catches_only_decoder_failure_and_is_limited() -> None:
    assert isinstance(MISSING, Missing)
    assert decode_output(
        response((0, "not-json")),
        TypeAdapter(list[str]).validate_json,
        default=["fallback", "extra", "ignored"],
        limit=2,
    ) == ["fallback", "extra"]
    assert decode_output(response((0, "not-an-integer")), int, default=None) is None
    with pytest.raises(DecodeError) as error:
        decode_output(response((0, "not-an-integer")), int)
    assert isinstance(error.value.__cause__, ValueError)


def test_pydantic_validation_error_is_reported_as_a_decode_error() -> None:
    with pytest.raises(DecodeError):
        decode_output(response((0, "not-json")), TypeAdapter(list[str]).validate_json)


def test_explicit_decode_error_uses_default_and_propagates_without_one() -> None:
    def rejecting_decoder(_text: str, /) -> str:
        msg = "decoder rejected output"
        raise DecodeError(msg)

    assert (
        decode_output(response((0, "value")), rejecting_decoder, default="fallback") == "fallback"
    )
    with pytest.raises(DecodeError, match="decoder rejected output"):
        decode_output(response((0, "value")), rejecting_decoder)


def test_default_does_not_hide_unexpected_decoder_exception() -> None:
    def exploding_decoder(_text: str, /) -> str:
        msg = "decoder bug"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="decoder bug"):
        decode_output(response((0, "value")), exploding_decoder, default="fallback")


def test_default_does_not_catch_output_selection_or_limiting_errors() -> None:
    calls: list[str] = []

    def recording_decoder(text: str, /) -> str:
        calls.append(text)
        return text

    with pytest.raises(IndexError):
        decode_output(response((0, "value")), recording_decoder, output_index=3, default="fallback")
    assert calls == []

    with pytest.raises(ValueError, match="greater than zero"):
        decode_output(response((0, "value")), recording_decoder, default="fallback", limit=0)


def test_collection_limiting_preserves_sequence_and_mapping_order() -> None:
    assert decode_output(
        response((0, "[1, 2, 3]")),
        TypeAdapter(list[int]).validate_json,
        limit=2,
    ) == [1, 2]
    assert decode_output(
        response((0, "[1, 2, 3]")),
        TypeAdapter(tuple[int, ...]).validate_json,
        limit=2,
    ) == (1, 2)
    assert decode_output(
        response((0, '{"first": 1, "second": 2, "third": 3}')),
        TypeAdapter(dict[str, int]).validate_json,
        limit=2,
    ) == {"first": 1, "second": 2}


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (set[int], {1, 2, 3}),
        (frozenset[int], frozenset({1, 2, 3})),
    ],
)
def test_collection_limiting_leaves_unordered_collections_unchanged(
    annotation: type[set[int]] | type[frozenset[int]],
    expected: set[int] | frozenset[int],
) -> None:
    decoder = TypeAdapter(annotation).validate_json

    assert decode_output(response((0, "[1, 2, 3]")), decoder, limit=1) == expected
