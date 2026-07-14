import base64
from collections.abc import Callable

import pytest
from pydantic import BaseModel, ConfigDict

from symai.operations import (
    BooleanMode,
    compare_request,
    data_uri,
    embedding_request,
    equals_request,
    image_request,
    language_request,
    limit_value,
    parse_boolean,
    parse_typed_output,
    parse_typed_output_with_metadata,
    parse_typed_value,
    summarize_request,
)
from symai.prompts import CompareValues, FuzzyEquals
from symai.runtime.models import (
    AssistantOutputMessage,
    EmbeddingRequest,
    FinishReason,
    ImageContent,
    ImageDetail,
    LanguageModelOutput,
    LanguageModelResponse,
    Provider,
    ResponseMetadata,
    SamplingConfig,
    SystemMessage,
    TextContent,
    UserMessage,
)

METADATA = ResponseMetadata(
    provider=Provider.OPENAI,
    model="gpt-5.4",
    status_code=200,
    request_id="request-id",
)


def _response(text: str) -> LanguageModelResponse:
    return LanguageModelResponse(
        outputs=(
            LanguageModelOutput(
                index=0,
                message=AssistantOutputMessage(content=(TextContent(text=text),)),
                finish_reason=FinishReason.STOP,
            ),
        ),
        metadata=METADATA,
    )


def test_language_request_builds_explicit_messages_examples_and_sampling() -> None:
    request = language_request(
        "Follow the instruction.",
        "question =>",
        examples=("first => answer", "second => answer"),
        max_tokens=64,
        stop=("END",),
    )

    assert request.messages == (
        SystemMessage(
            content=(
                TextContent(text="Follow the instruction.\nfirst => answer\nsecond => answer"),
            )
        ),
        UserMessage(content=(TextContent(text="question =>"),)),
    )
    assert request.sampling == SamplingConfig(max_tokens=64, stop=("END",))


@pytest.mark.parametrize(
    ("field", "call"),
    [
        ("examples", lambda: language_request("system", "user", examples="example")),
        ("stop", lambda: language_request("system", "user", stop="END")),
        ("inputs", lambda: embedding_request("input")),
    ],
)
def test_string_sequences_are_rejected_instead_of_splitting_into_characters(
    field: str,
    call: Callable[[], object],
) -> None:
    with pytest.raises(TypeError, match=field):
        call()


def test_representative_prompt_builders_preserve_visible_prompt_text() -> None:
    summary = summarize_request("A long text", context="Research")
    equals = equals_request("one", 1)
    comparison = compare_request(4, ">", 3)

    assert summary.messages == (
        SystemMessage(
            content=(TextContent(text="Summarize the content of the following text:\n"),)
        ),
        UserMessage(content=(TextContent(text="Context: Research Text: A long text\n"),)),
    )
    equals_system = "\n".join(
        (
            "Make a fuzzy equals comparison; are the following objects contextually the same?\n",
            *FuzzyEquals().value,
        )
    )
    assert equals.messages == (
        SystemMessage(content=(TextContent(text=equals_system),)),
        UserMessage(content=(TextContent(text="one == 1 =>"),)),
    )
    comparison_system = "\n".join(
        ("Compare 'A' and 'B' based on the operator:\n", *CompareValues().value)
    )
    assert comparison.messages == (
        SystemMessage(content=(TextContent(text=comparison_system),)),
        UserMessage(content=(TextContent(text="4 > 3 =>"),)),
    )


@pytest.mark.parametrize(
    ("value", "mode", "expected"),
    [
        ("True", BooleanMode.STRICT, True),
        ("yes", BooleanMode.MEDIUM, True),
        ("1", BooleanMode.MEDIUM, False),
        ("1", BooleanMode.TOLERANT, True),
        ("certainly", BooleanMode.TOLERANT, True),
        ("false", BooleanMode.TOLERANT, False),
    ],
)
def test_parse_boolean_uses_explicit_tolerance(
    value: str, mode: BooleanMode, expected: bool
) -> None:
    assert parse_boolean(value, mode=mode) is expected


def test_parse_typed_value_handles_scalars_collections_and_pydantic_models() -> None:
    class Result(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")

        answer: int

    assert parse_typed_value("42", int) == 42
    assert parse_typed_value("[1, 2]", list) == [1, 2]
    assert parse_typed_value("{'a': 1}", dict) == {"a": 1}
    assert parse_typed_value('{"answer": 7}', Result) == Result(answer=7)

    with pytest.raises(ValueError, match="Expected list"):
        parse_typed_value("{'a': 1}", list)


def test_typed_output_applies_defaults_limits_and_returns_normalized_metadata() -> None:
    response = _response("['one', 'two', 'three']")

    assert parse_typed_output(response, list, limit=2) == ["one", "two"]
    parsed, metadata = parse_typed_output_with_metadata(response, list, limit=1)
    assert parsed == ["one"]
    assert metadata is METADATA

    assert parse_typed_output(_response("not-an-integer"), int, default=9) == 9
    assert parse_typed_output(
        _response("not-a-list"),
        list,
        default=["one", "two", "three"],
        limit=2,
    ) == ["one", "two"]
    with pytest.raises(ValueError, match="invalid literal"):
        parse_typed_output(_response("not-an-integer"), int)


def test_typed_output_selects_the_explicit_normalized_index() -> None:
    response = LanguageModelResponse(
        outputs=(
            LanguageModelOutput(
                index=1,
                message=AssistantOutputMessage(content=(TextContent(text="8"),)),
                finish_reason=FinishReason.STOP,
            ),
            LanguageModelOutput(
                index=0,
                message=AssistantOutputMessage(content=(TextContent(text="7"),)),
                finish_reason=FinishReason.STOP,
            ),
        ),
        metadata=METADATA,
    )

    assert parse_typed_output(response, int, index=1) == 8
    with pytest.raises(IndexError, match="output index 2"):
        parse_typed_output(response, int, index=2)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2, 3], [1, 2]),
        ((1, 2, 3), (1, 2)),
        ({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2}),
        ("unchanged", "unchanged"),
    ],
)
def test_limit_value_is_deterministic(value: object, expected: object) -> None:
    assert limit_value(value, 2) == expected


def test_image_and_data_uri_preparation_are_pure() -> None:
    payload = b"\x89PNG\r\n"
    encoded = data_uri(payload, "image/png")

    assert encoded == f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"
    request = image_request(
        "Describe the image.",
        "Focus on the foreground.",
        image_url=encoded,
        detail=ImageDetail.HIGH,
    )
    assert request.messages == (
        SystemMessage(content=(TextContent(text="Describe the image."),)),
        UserMessage(
            content=(
                TextContent(text="Focus on the foreground."),
                ImageContent(url=encoded, detail=ImageDetail.HIGH),
            )
        ),
    )

    with pytest.raises(ValueError, match="media_type"):
        data_uri(payload, "image/png;unsafe")


def test_embedding_request_uses_immutable_normalized_inputs() -> None:
    inputs = ["first", "second"]

    request = embedding_request(inputs, dimensions=256, user="customer")
    inputs.append("later mutation")

    assert request == EmbeddingRequest(
        inputs=("first", "second"),
        dimensions=256,
        user="customer",
    )
