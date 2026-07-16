import json
from math import inf, nan
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

import symai.runtime.models as models
from symai.runtime.models import (
    AssistantMessage,
    AssistantOutputMessage,
    Content,
    DeveloperMessage,
    EmbeddingModelSpec,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FinishReason,
    ImageContent,
    ImageDetail,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    LanguageModelSpec,
    Message,
    MetadataLabel,
    RateLimitMetadata,
    ReasoningConfig,
    ReasoningEffort,
    ReasoningField,
    ReasoningFormat,
    ReasoningSummary,
    ResponseFormat,
    ResponseMetadata,
    SamplingConfig,
    SamplingField,
    SystemMessage,
    TextContent,
    TextResponseFormat,
    TokenUsage,
    UserMessage,
)


def test_public_models_are_strict_frozen_and_forbid_extra_fields():
    with pytest.raises(ValidationError):
        TextContent.model_validate({"text": 1})
    with pytest.raises(ValidationError):
        TextContent.model_validate({"text": "hello", "unexpected": True})

    content = TextContent(text="hello")
    with pytest.raises(ValidationError):
        content.text = "changed"


def test_json_schema_response_format_round_trips_plain_json_values():
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "tags": {"type": "array", "items": [{"type": "string"}]},
        },
        "required": ["answer"],
    }

    response_format = JsonSchemaResponseFormat(
        name="answer",
        json_schema=schema,
        strict=True,
    )

    assert response_format.json_schema == schema
    assert response_format.model_dump(mode="json")["json_schema"] == schema


def test_json_schema_response_format_rejects_non_json_values():
    with pytest.raises(ValidationError):
        JsonSchemaResponseFormat.model_validate(
            {
                "name": "answer",
                "json_schema": {"value": object()},
                "strict": True,
            }
        )


def test_content_message_and_response_format_unions_are_discriminated():
    content_adapter = TypeAdapter(Content)
    message_adapter = TypeAdapter(Message)
    response_format_adapter = TypeAdapter(ResponseFormat)

    assert isinstance(
        content_adapter.validate_python({"type": "text", "text": "hello"}), TextContent
    )
    assert isinstance(
        content_adapter.validate_python(
            {"type": "image", "url": "data:image/png;base64,AA==", "detail": ImageDetail.HIGH}
        ),
        ImageContent,
    )
    assert isinstance(
        message_adapter.validate_python(
            {"role": "developer", "content": ({"type": "text", "text": "rules"},)}
        ),
        DeveloperMessage,
    )
    assert isinstance(
        response_format_adapter.validate_python({"type": "json_object"}),
        JsonObjectResponseFormat,
    )

    with pytest.raises(ValidationError):
        message_adapter.validate_python({"role": "unknown", "content": ()})
    with pytest.raises(ValidationError):
        response_format_adapter.validate_python({"type": "yaml"})


def test_role_specific_messages_accept_only_their_content_contracts():
    text = TextContent(text="hello")
    image = ImageContent(url="https://example.com/image.png")

    assert SystemMessage(content=(text,)).role == "system"
    assert DeveloperMessage(content=(text,)).role == "developer"
    assert UserMessage(content=(text, image)).role == "user"

    with pytest.raises(ValidationError):
        SystemMessage.model_validate({"content": (image,)})
    with pytest.raises(ValidationError):
        UserMessage(content=())


def test_assistant_message_requires_content_or_reasoning():
    text = TextContent(text="answer")

    assert AssistantMessage(content=(text,)).content == (text,)
    assert AssistantMessage(reasoning=text).reasoning == text
    with pytest.raises(ValidationError):
        AssistantMessage()


def test_json_schema_response_format_uses_non_colliding_public_name():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    response_format = JsonSchemaResponseFormat(name="answer", json_schema=schema, strict=True)

    assert response_format.type == "json_schema"
    assert response_format.json_schema == schema
    assert callable(JsonSchemaResponseFormat.schema)
    assert "schema" not in JsonSchemaResponseFormat.model_fields
    assert response_format.model_dump()["json_schema"] == schema
    assert "json_schema" in json.loads(response_format.model_dump_json())


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("max_tokens", 0),
        ("temperature", -0.1),
        ("temperature", inf),
        ("top_p", 1.1),
        ("frequency_penalty", -2.1),
        ("presence_penalty", 2.1),
    ],
)
def test_sampling_numeric_fields_are_finite_and_bounded(field, invalid):
    with pytest.raises(ValidationError):
        SamplingConfig(**{field: invalid})


def test_sampling_and_reasoning_config_are_typed_and_deeply_frozen():
    sampling = SamplingConfig(
        max_tokens=100,
        temperature=0.5,
        top_p=0.9,
        stop=("done",),
        seed=7,
        frequency_penalty=-0.5,
        presence_penalty=0.5,
    )
    reasoning = ReasoningConfig(
        enabled=True,
        effort=ReasoningEffort.HIGH,
        summary=ReasoningSummary.CONCISE,
        format=ReasoningFormat.PARSED,
        clear=True,
    )

    assert isinstance(sampling.stop, tuple)
    assert reasoning.enabled is True
    with pytest.raises(ValidationError):
        sampling.stop = ("changed",)
    with pytest.raises(ValidationError):
        reasoning.clear = False


def test_requests_are_concrete_frozen_and_raw_payload_free():
    message = UserMessage(content=(TextContent(text="hello"),))
    language_request = LanguageModelRequest(
        messages=(message,), metadata=(MetadataLabel(key="trace", value="abc"),)
    )
    embedding_request = EmbeddingRequest(inputs=("hello",), dimensions=128)

    assert isinstance(language_request.response_format, TextResponseFormat)
    assert isinstance(language_request.sampling, SamplingConfig)
    assert "raw" not in language_request.__class__.model_fields
    assert "raw" not in embedding_request.__class__.model_fields
    assert not get_args(language_request.__class__)
    assert not get_args(embedding_request.__class__)

    with pytest.raises(ValidationError):
        LanguageModelRequest(messages=())
    with pytest.raises(ValidationError):
        EmbeddingRequest(inputs=())
    with pytest.raises(ValidationError):
        EmbeddingRequest(inputs=("hello",), dimensions=0)


def test_language_request_metadata_preserves_order_and_rejects_duplicate_keys():
    message = UserMessage(content=(TextContent(text="hello"),))
    first = MetadataLabel(key="trace", value="abc")
    second = MetadataLabel(key="tenant", value="one")

    request = LanguageModelRequest(messages=(message,), metadata=(first, second))

    assert request.metadata == (first, second)
    with pytest.raises(ValidationError):
        LanguageModelRequest(
            messages=(message,),
            metadata=(first, MetadataLabel(key="trace", value="duplicate")),
        )


def test_usage_defaults_to_zero_and_rejects_negative_counts():
    usage = TokenUsage()

    assert usage.model_dump() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_prompt_tokens": 0,
        "cache_miss_prompt_tokens": 0,
        "reasoning_tokens": 0,
        "image_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }
    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1)


def test_language_output_text_concatenates_ordered_parts_without_delimiters():
    output = LanguageModelOutput(
        index=0,
        message=AssistantOutputMessage(
            content=(TextContent(text="first"), TextContent(text=" second"))
        ),
        finish_reason=FinishReason.STOP,
    )
    metadata = ResponseMetadata(
        provider="openai",
        requested_model="gpt-5.5",
        status_code=200,
    )
    response = LanguageModelResponse(outputs=(output,), metadata=metadata)

    assert output.text == "first second"
    assert response.outputs == (output,)


def test_language_output_supports_refusal_without_invented_assistant_text():
    output = LanguageModelOutput(
        index=0,
        message=AssistantOutputMessage(),
        refusal="I cannot help with that.",
        finish_reason=FinishReason.STOP,
    )

    assert output.text == ""
    assert output.refusal == "I cannot help with that."

    with pytest.raises(ValidationError):
        LanguageModelOutput(
            index=0,
            message=AssistantOutputMessage(),
            finish_reason=FinishReason.STOP,
        )


def test_language_output_allows_empty_only_for_terminal_content_filter():
    output = LanguageModelOutput(
        index=0,
        message=AssistantOutputMessage(),
        finish_reason=FinishReason.CONTENT_FILTER,
    )

    assert output.text == ""
    assert output.message.content == ()
    assert output.refusal is None


@pytest.mark.parametrize(
    "message",
    [
        AssistantOutputMessage(content=(TextContent(text=""),)),
        AssistantOutputMessage(reasoning=TextContent(text="")),
    ],
)
def test_successful_output_rejects_empty_content_and_reasoning(message):
    with pytest.raises(ValidationError):
        LanguageModelOutput(
            index=0,
            message=message,
            finish_reason=FinishReason.STOP,
        )


def test_embedding_vectors_require_non_negative_indices_and_finite_values():
    metadata = ResponseMetadata(
        provider="openai",
        requested_model="embedding",
        status_code=200,
    )
    vector = EmbeddingVector(index=0, values=(0.1, -0.2))
    response = EmbeddingResponse(vectors=(vector,), metadata=metadata)

    assert response.vectors == (vector,)
    with pytest.raises(ValidationError):
        EmbeddingVector(index=-1, values=(0.1,))
    with pytest.raises(ValidationError):
        EmbeddingVector(index=0, values=())
    with pytest.raises(ValidationError):
        EmbeddingVector(index=0, values=(nan,))


def test_response_metadata_and_rate_limits_are_frozen_and_bounded():
    rate_limit = RateLimitMetadata(
        limit_requests_day=100,
        limit_tokens_minute=1_000,
        remaining_requests_day=99,
        remaining_tokens_minute=900,
        reset_requests_day=30.5,
        reset_tokens_minute=5.5,
    )
    metadata = ResponseMetadata(
        provider="cerebras",
        requested_model="gpt-oss-120b",
        response_model="provider-resolved-cerebras-model",
        status_code=200,
        retry_after=0.0,
        created_at=1.0,
        usage=TokenUsage(total_tokens=1),
        rate_limit=rate_limit,
    )

    assert metadata.rate_limit is rate_limit
    assert metadata.requested_model == "gpt-oss-120b"
    assert metadata.response_model == "provider-resolved-cerebras-model"
    assert "model" not in metadata.model_dump()
    with pytest.raises(ValidationError):
        ResponseMetadata.model_validate({"provider": "openai", "model": "gpt", "status_code": 200})
    with pytest.raises(ValidationError):
        metadata.status_code = 500
    with pytest.raises(ValidationError):
        ResponseMetadata(provider="openai", requested_model="gpt", status_code=99)
    with pytest.raises(ValidationError):
        RateLimitMetadata(reset_tokens_minute=inf)


def test_response_metadata_accepts_open_provider_ids_and_normalizes_case() -> None:
    metadata = ResponseMetadata(
        provider="ACME_Local",
        requested_model="offline-model",
        status_code=200,
    )

    assert metadata.provider == "acme_local"
    with pytest.raises(ValidationError):
        ResponseMetadata(
            provider="acme local",
            requested_model="offline-model",
            status_code=200,
        )


def test_model_feature_metadata_contains_only_authoritative_capabilities():
    language_spec = LanguageModelSpec(
        response_tokens=32_000,
        reasoning_fields=(ReasoningField.EFFORT,),
        reasoning_efforts=(ReasoningEffort.HIGH,),
        sampling_fields=(SamplingField.MAX_TOKENS, SamplingField.TOP_P),
        vision=False,
    )
    embedding_spec = EmbeddingModelSpec(dimensions=1_536)

    assert language_spec.reasoning_efforts == (ReasoningEffort.HIGH,)
    assert embedding_spec.dimensions == 1_536
    assert {
        "context_tokens",
        "message_roles",
        "content_types",
        "response_formats",
    }.isdisjoint(LanguageModelSpec.model_fields)
    assert "context_tokens" not in EmbeddingModelSpec.model_fields
    assert {
        "logprobs",
        "top_logprobs",
        "logit_bias",
    }.isdisjoint(SamplingConfig.model_fields)
    assert {
        "JsonEntry",
        "JsonArray",
        "JsonObject",
        "MessageRole",
        "ContentType",
        "ResponseFormatType",
        "LogitBias",
    }.isdisjoint(vars(models))
    assert {
        "logprobs",
        "top_logprobs",
        "logit_bias",
    }.isdisjoint(field.value for field in SamplingField)

    with pytest.raises(ValidationError):
        SamplingConfig.model_validate({"logprobs": True})
