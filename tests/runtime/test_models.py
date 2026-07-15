import json
from math import inf, nan
from typing import get_args

import pytest
from pydantic import SecretStr, TypeAdapter, ValidationError

from symai.runtime.models import (
    AssistantMessage,
    AssistantOutputMessage,
    Content,
    ContentType,
    DeveloperMessage,
    EmbeddingModelSpec,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FinishReason,
    ImageContent,
    ImageDetail,
    JsonArray,
    JsonEntry,
    JsonObject,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    LanguageModelSpec,
    LogitBias,
    Message,
    MessageRole,
    MetadataLabel,
    Provider,
    ProviderEngineConfig,
    RateLimitMetadata,
    ReasoningConfig,
    ReasoningEffort,
    ReasoningField,
    ReasoningFormat,
    ReasoningSummary,
    ResponseFormat,
    ResponseFormatType,
    ResponseMetadata,
    RuntimeConfig,
    SamplingConfig,
    SamplingField,
    SystemMessage,
    TextContent,
    TextResponseFormat,
    TokenUsage,
    TransportConfig,
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


def test_json_boundary_recursively_freezes_and_serializes_values():
    source = {
        "name": "Ada",
        "active": True,
        "score": 2.5,
        "attributes": {"level": 3, "empty": None},
        "tags": ["math", {"year": 1843}],
    }

    value = JsonObject.parse(source)

    assert isinstance(value.entries, tuple)
    attributes = value.entries[3].value
    tags = value.entries[4].value
    assert isinstance(attributes, JsonObject)
    assert isinstance(tags, JsonArray)
    assert isinstance(tags.values, tuple)
    assert isinstance(tags.values[1], JsonObject)
    assert value.to_builtin() == source

    with pytest.raises(ValidationError):
        value.entries = ()
    with pytest.raises(ValidationError):
        tags.values = ()


@pytest.mark.parametrize("invalid", [nan, inf, -inf, {"bad": object()}])
def test_json_boundary_rejects_non_json_values(invalid):
    with pytest.raises((TypeError, ValueError, ValidationError)):
        JsonObject.parse({"value": invalid})


def test_json_object_rejects_duplicate_direct_entries():
    duplicate_entries = (
        JsonEntry(key="same", value=1),
        JsonEntry(key="same", value=2),
    )

    with pytest.raises(ValidationError):
        JsonObject(entries=duplicate_entries)


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
    schema_value = {"type": "object", "properties": {"answer": {"type": "string"}}}
    schema = JsonObject.parse(schema_value)
    response_format = JsonSchemaResponseFormat(name="answer", json_schema=schema, strict=True)

    assert response_format.type == "json_schema"
    assert response_format.json_schema is schema
    assert callable(JsonSchemaResponseFormat.schema)
    assert "schema" not in JsonSchemaResponseFormat.model_fields
    assert response_format.model_dump()["json_schema"] == schema.model_dump()
    assert "json_schema" in json.loads(response_format.model_dump_json())
    assert response_format.json_schema.to_builtin() == schema_value


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("max_tokens", 0),
        ("temperature", -0.1),
        ("temperature", inf),
        ("top_p", 1.1),
        ("frequency_penalty", -2.1),
        ("presence_penalty", 2.1),
        ("top_logprobs", 21),
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
        logprobs=True,
        top_logprobs=5,
        logit_bias=(LogitBias(token="42", value=-10.0),),
    )
    reasoning = ReasoningConfig(
        enabled=True,
        effort=ReasoningEffort.HIGH,
        summary=ReasoningSummary.CONCISE,
        format=ReasoningFormat.PARSED,
        clear=True,
    )

    assert isinstance(sampling.stop, tuple)
    assert isinstance(sampling.logit_bias, tuple)
    assert reasoning.enabled is True
    with pytest.raises(ValidationError):
        sampling.stop = ("changed",)
    with pytest.raises(ValidationError):
        reasoning.clear = False


@pytest.mark.parametrize("value", [nan, inf, -inf, -100.1, 100.1])
def test_logit_bias_rejects_non_finite_or_out_of_range_values(value):
    with pytest.raises(ValidationError):
        LogitBias(token="1", value=value)


def test_sampling_logit_bias_preserves_order_and_rejects_duplicate_tokens():
    first = LogitBias(token="42", value=-10.0)
    second = LogitBias(token="7", value=5.0)

    sampling = SamplingConfig(logit_bias=(first, second))

    assert sampling.logit_bias == (first, second)
    with pytest.raises(ValidationError):
        SamplingConfig(
            logit_bias=(first, LogitBias(token="42", value=20.0)),
        )


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
        provider=Provider.OPENAI,
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
        provider=Provider.OPENAI,
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
        provider=Provider.CEREBRAS,
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
        ResponseMetadata(provider=Provider.OPENAI, model="gpt", status_code=200)
    with pytest.raises(ValidationError):
        metadata.status_code = 500
    with pytest.raises(ValidationError):
        ResponseMetadata(provider=Provider.OPENAI, requested_model="gpt", status_code=99)
    with pytest.raises(ValidationError):
        RateLimitMetadata(reset_tokens_minute=inf)


def test_runtime_configuration_is_frozen_nonempty_and_redacts_api_keys():
    transport = TransportConfig(request_timeout=30.0, connect_timeout=2.0, connect_retries=1)
    engine = ProviderEngineConfig(
        provider=Provider.DEEPSEEK,
        model="deepseek-v4-flash",
        api_key=SecretStr("top-secret"),
        transport=transport,
    )
    config = RuntimeConfig(language_model=engine)

    assert engine.api_key.get_secret_value() == "top-secret"
    assert "top-secret" not in repr(engine)
    assert "top-secret" not in engine.model_dump_json()
    with pytest.raises(ValidationError):
        config.language_model = None
    with pytest.raises(ValidationError):
        RuntimeConfig()
    with pytest.raises(ValidationError):
        TransportConfig(request_timeout=inf)
    with pytest.raises(ValidationError):
        TransportConfig(connect_retries=-1)


@pytest.mark.parametrize("api_key", ["", SecretStr("")])
def test_provider_engine_configuration_rejects_empty_api_keys(api_key):
    with pytest.raises(ValidationError):
        ProviderEngineConfig(
            provider=Provider.OPENAI,
            model="gpt-5.5",
            api_key=api_key,
        )


def test_model_feature_metadata_is_strict_frozen_and_typed():
    language_spec = LanguageModelSpec(
        context_tokens=128_000,
        response_tokens=32_000,
        message_roles=(MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT),
        content_types=(ContentType.TEXT,),
        response_formats=(ResponseFormatType.TEXT, ResponseFormatType.JSON_OBJECT),
        reasoning_fields=(ReasoningField.EFFORT,),
        reasoning_efforts=(ReasoningEffort.HIGH,),
        sampling_fields=(SamplingField.MAX_TOKENS, SamplingField.TOP_P),
        vision=False,
    )
    embedding_spec = EmbeddingModelSpec(context_tokens=8_191, dimensions=1_536)

    assert language_spec.reasoning_efforts == (ReasoningEffort.HIGH,)
    assert embedding_spec.dimensions == 1_536
    with pytest.raises(ValidationError):
        language_spec.context_tokens = 1
    with pytest.raises(ValidationError):
        LanguageModelSpec(
            context_tokens=0,
            response_tokens=1,
            message_roles=(MessageRole.USER,),
            content_types=(ContentType.TEXT,),
            response_formats=(ResponseFormatType.TEXT,),
        )
