import base64

import pytest

from symai.operations import (
    data_uri,
    embedding_request,
    image_request,
    language_request,
    parse_embedding_response,
)
from symai.runtime.models import (
    EmbeddingResponse,
    EmbeddingVector,
    ImageContent,
    ImageDetail,
    LanguageModelResponse,
    ResponseMetadata,
    SamplingConfig,
    SystemMessage,
    TextContent,
    UserMessage,
)

METADATA = ResponseMetadata(
    provider="test",
    requested_model="test-model",
    status_code=200,
    request_id="request-1",
)


def test_language_request_builds_immutable_normalized_messages() -> None:
    examples = ["one", "two"]
    stops = ["END"]

    request = language_request(
        "System",
        "User",
        examples=examples,
        max_tokens=64,
        stop=stops,
    )
    examples.append("changed")
    stops.append("changed")

    assert request.messages == (
        SystemMessage(content=(TextContent(text="System\none\ntwo"),)),
        UserMessage(content=(TextContent(text="User"),)),
    )
    assert request.sampling == SamplingConfig(max_tokens=64, stop=("END",))


@pytest.mark.parametrize(
    ("call", "field"),
    [
        (lambda: language_request("", "", examples="abc"), "examples"),
        (lambda: language_request("", "", stop="abc"), "stop"),
        (lambda: embedding_request("abc"), "inputs"),
    ],
)
def test_normalized_sequence_helpers_reject_one_string(call: object, field: str) -> None:
    with pytest.raises(TypeError, match=field):
        call()  # type: ignore[operator]


def test_image_request_preserves_multimodal_content_and_sampling() -> None:
    request = image_request(
        "Inspect the image",
        "What is shown?",
        image_url="data:image/png;base64,AA==",
        detail=ImageDetail.HIGH,
        max_tokens=32,
        stop=("DONE",),
    )

    assert request.messages == (
        SystemMessage(content=(TextContent(text="Inspect the image"),)),
        UserMessage(
            content=(
                TextContent(text="What is shown?"),
                ImageContent(url="data:image/png;base64,AA==", detail=ImageDetail.HIGH),
            )
        ),
    )
    assert request.sampling == SamplingConfig(max_tokens=32, stop=("DONE",))


def test_data_uri_encodes_bytes_and_rejects_invalid_media_type() -> None:
    payload = b"\x89PNG\r\n"

    assert data_uri(payload, "image/png") == (
        f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"
    )
    with pytest.raises(ValueError, match="media_type"):
        data_uri(payload, "image/png;unsafe")


def test_embedding_request_snapshots_normalized_inputs() -> None:
    inputs = ["first", "second"]

    request = embedding_request(inputs, dimensions=8, user="tenant-user")
    inputs.append("changed")

    assert request.inputs == ("first", "second")
    assert request.dimensions == 8
    assert request.user == "tenant-user"


def test_embedding_parser_orders_provider_vectors_and_rejects_duplicates() -> None:
    response = EmbeddingResponse(
        vectors=(
            EmbeddingVector(index=1, values=(3.0, 4.0)),
            EmbeddingVector(index=0, values=(1.0, 2.0)),
        ),
        metadata=METADATA,
    )

    assert parse_embedding_response(response) == ((1.0, 2.0), (3.0, 4.0))

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


def test_helper_module_has_no_semantic_operation_request_builders() -> None:
    import symai.operations as operations

    removed = {
        "combine_request",
        "compare_request",
        "contains_request",
        "contextualize_language_request",
        "convert_request",
        "endswith_request",
        "equals_request",
        "extract_request",
        "filter_request",
        "getitem_request",
        "include_request",
        "interpret_request",
        "invert_request",
        "isinstanceof_request",
        "logic_request",
        "map_request",
        "modify_request",
        "negate_request",
        "query_request",
        "rank_request",
        "replace_request",
        "setitem_request",
        "startswith_request",
        "style_request",
        "summarize_request",
        "translate_request",
    }
    assert removed.isdisjoint(vars(operations))


def test_helper_annotations_remain_provider_neutral() -> None:
    assert LanguageModelResponse.__module__ == "symai.runtime.models"
