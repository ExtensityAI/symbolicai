import base64
from collections.abc import Callable

import pytest

from symai.operations import (
    combine_request,
    compare_request,
    contains_request,
    convert_request,
    data_uri,
    embedding_request,
    endswith_request,
    equals_request,
    extract_request,
    filter_request,
    getitem_request,
    image_request,
    include_request,
    interpret_request,
    invert_request,
    isinstanceof_request,
    language_request,
    logic_request,
    map_request,
    modify_request,
    negate_request,
    parse_embedding_response,
    query_request,
    rank_request,
    replace_request,
    setitem_request,
    startswith_request,
    style_request,
    summarize_request,
    translate_request,
)
from symai.prompts import (
    CombineText,
    CompareValues,
    ContainsValue,
    EndsWith,
    ExtractPattern,
    Filter,
    Format,
    FuzzyEquals,
    IncludeText,
    Index,
    InvertExpression,
    IsInstanceOf,
    LogicExpression,
    MapExpression,
    Modify,
    NegateStatement,
    RankList,
    RemoveIndex,
    ReplaceText,
    SetIndex,
    SimpleSymbolicExpression,
    StartsWith,
)
from symai.runtime.models import (
    AssistantOutputMessage,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FinishReason,
    ImageContent,
    ImageDetail,
    LanguageModelOutput,
    LanguageModelResponse,
    ResponseMetadata,
    SamplingConfig,
    SystemMessage,
    TextContent,
    UserMessage,
)

METADATA = ResponseMetadata(
    provider="openai",
    requested_model="gpt-5.4",
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


@pytest.mark.parametrize(
    ("normalized_request", "expected"),
    [
        (
            contains_request("container", "item"),
            language_request(
                "Is semantically the information of 'A' contained in 'B'?\n",
                "item in container =>",
                examples=ContainsValue().value,
            ),
        ),
        (
            startswith_request("value", "pre"),
            language_request(
                "Does 'A' start with 'B'?\n",
                "value startswith pre =>",
                examples=StartsWith().value,
            ),
        ),
        (
            endswith_request("value", "post"),
            language_request(
                "Does 'A' end with 'B'?\n",
                "value endswith post =>",
                examples=EndsWith().value,
            ),
        ),
        (
            isinstanceof_request("value", "kind"),
            language_request(
                "Is 'A' an instance of 'B'?\n",
                "value isinstanceof kind =>",
                examples=IsInstanceOf().value,
            ),
        ),
        (
            negate_request("statement"),
            language_request(
                "Negate the following statement:\n",
                "statement =>",
                examples=NegateStatement().value,
            ),
        ),
        (
            invert_request("statement"),
            language_request(
                "Invert the logic of the content:\n",
                "statement =>",
                examples=InvertExpression().value,
            ),
        ),
        (
            getitem_request("value", "key"),
            language_request(
                "Get item at index position:\n",
                "value index key =>",
                examples=Index().value,
            ),
        ),
        (
            setitem_request("value", "key", "replacement"),
            language_request(
                "Set item at index position:\n",
                "value index key set replacement =>",
                examples=SetIndex().value,
            ),
        ),
        (
            setitem_request("value", "key", None, delete=True),
            language_request(
                "Delete the items at the index position:\n",
                "value remove key =>",
                examples=RemoveIndex().value,
            ),
        ),
        (
            modify_request("value", "changes"),
            language_request(
                "Modify the text to match the criteria:\n",
                "text 'value' modify 'changes'=>",
                examples=Modify().value,
            ),
        ),
        (
            filter_request("value", "criteria", include=True),
            language_request(
                "Filter the information from the text based on the filter criteria. "
                "Leave sentences unchanged if they are unrelated to the filter criteria:\n",
                "text 'value' include 'criteria' =>",
                examples=Filter().value,
            ),
        ),
        (
            map_request(["one"], "uppercase"),
            language_request(
                "Transform each element in the input based on the instruction. "
                "Preserve container type and elements that don't match the instruction:\n",
                "text '['one']' uppercase =>",
                examples=MapExpression().value,
            ),
        ),
        (
            convert_request("value", "JSON"),
            language_request(
                "Translate the following text into JSON format.\n",
                "text value format 'JSON' =>",
                examples=Format().value,
            ),
        ),
        (
            extract_request("value", "pattern"),
            language_request(
                "Extract a pattern from text:\n",
                "from 'value' extract 'pattern' =>",
                examples=ExtractPattern().value,
            ),
        ),
        (
            interpret_request("1 + 1", prompt="Evaluate:\n"),
            language_request(
                "Evaluate:\n",
                "1 + 1 =>",
                examples=SimpleSymbolicExpression().value,
            ),
        ),
        (
            logic_request("left", "and", "right"),
            language_request(
                "Evaluate the logic expressions:\n",
                "expr :left: and :right: =>",
                examples=LogicExpression().value,
            ),
        ),
        (
            replace_request("value", "old", "new"),
            language_request(
                "Replace text parts by string pattern.\n",
                "text 'value' replace 'old' with 'new'=>",
                examples=ReplaceText().value,
            ),
        ),
        (
            include_request("value", "information"),
            language_request(
                "Include information based on description.\n",
                "text 'value' include 'information' =>",
                examples=IncludeText().value,
            ),
        ),
        (
            combine_request("left", "right"),
            language_request(
                "Add the two data types in a logical way:\n",
                "left + right =>",
                examples=CombineText().value,
            ),
        ),
        (
            rank_request("one | two", "quality", order="asc"),
            language_request(
                "Order the list of objects based on their quality measure and oder literal:\n",
                "order: 'asc' measure: 'quality' list: ['one', 'two'] =>",
                examples=RankList().value,
            ),
        ),
        (
            translate_request("hello", "German"),
            language_request(
                "Your task is to translate and **only** translate the text into German:\n",
                "hello",
            ),
        ),
        (
            style_request("value", "minimal", libraries=("css", "html")),
            language_request(
                "Style the [DATA] based on best practices and the descriptions in [...] brackets. "
                "Do not remove content from the data! Do not add libraries or other descriptions. \n",
                "[FORMAT]: minimal\n[LIBRARIES]: css, html\n[DATA]:\nvalue\n\n",
            ),
        ),
        (
            query_request("facts", "question"),
            language_request("", "Data:\nfacts\nContext: question\nAnswer:"),
        ),
    ],
)
def test_operation_specific_builders_preserve_normalized_request_contract(
    normalized_request: object,
    expected: object,
) -> None:
    assert normalized_request == expected


def test_embedding_parser_is_normalized_and_deterministic() -> None:
    response = EmbeddingResponse(
        vectors=(
            EmbeddingVector(index=1, values=(3.0, 4.0)),
            EmbeddingVector(index=0, values=(1.0, 2.0)),
        ),
        metadata=METADATA,
    )
    assert parse_embedding_response(response) == [[1.0, 2.0], [3.0, 4.0]]

    duplicate = response.model_copy(
        update={
            "vectors": (
                EmbeddingVector(index=0, values=(1.0,)),
                EmbeddingVector(index=0, values=(2.0,)),
            )
        }
    )
    with pytest.raises(ValueError, match="indices must be unique"):
        parse_embedding_response(duplicate)
