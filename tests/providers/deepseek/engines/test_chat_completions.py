import json
from collections.abc import Callable
from typing import cast

import httpx
import pytest
from pydantic import JsonValue, SecretStr, ValidationError

from symai.providers.deepseek import ChatCompletionsEngine
from symai.providers.deepseek.client import errors as deepseek_errors
from symai.providers.deepseek.client import Client
from symai.providers.deepseek.client.transport import ResponseMetadata as DeepSeekResponseMetadata
from symai.runtime.errors import (
    AuthenticationError,
    ExecutionError,
    InvalidResponseError,
    RateLimitError,
    TransportError,
    UnsupportedFeatureError,
    UnsupportedModelError,
)
from symai.runtime.models import (
    AssistantMessage,
    DeveloperMessage,
    FinishReason,
    ImageContent,
    JsonObject,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    LanguageModelRequest,
    LogitBias,
    MetadataLabel,
    ReasoningConfig,
    ReasoningEffort,
    ReasoningFormat,
    ReasoningSummary,
    SamplingConfig,
    SystemMessage,
    TextContent,
    UserMessage,
)


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Client:
    return Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )


def _choice(
    *,
    index: int = 0,
    finish_reason: str = "stop",
    content: str | None = "answer",
    reasoning_content: str | None = "thought",
    role: str = "assistant",
) -> dict[str, JsonValue]:
    return {
        "finish_reason": finish_reason,
        "index": index,
        "message": {
            "role": role,
            "content": content,
            "reasoning_content": reasoning_content,
        },
    }


def _chat_json(
    *,
    choices: list[dict[str, JsonValue]] | None = None,
    usage: dict[str, JsonValue] | None = None,
    model: str = "deepseek-v4-flash",
    object_type: str = "chat.completion",
) -> dict[str, object]:
    return {
        "id": "response-id",
        "choices": choices if choices is not None else [_choice()],
        "created": 10,
        "model": model,
        "object": object_type,
        "system_fingerprint": "fingerprint",
        "usage": usage
        if usage is not None
        else {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_cache_hit_tokens": 3,
            "prompt_cache_miss_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }


def test_model_catalog_exposes_normalized_deepseek_capabilities() -> None:
    engine = ChatCompletionsEngine(client=cast("Client", object()), model="deepseek-v4-flash")

    assert engine.provider == "deepseek"
    assert engine.model == "deepseek-v4-flash"
    assert engine.model_spec.context_tokens == 1_000_000
    assert engine.model_spec.response_tokens == 384_000
    assert engine.model_spec.vision is False
    assert tuple(role.value for role in engine.model_spec.message_roles) == (
        "system",
        "user",
        "assistant",
    )
    assert tuple(value.value for value in engine.model_spec.response_formats) == (
        "text",
        "json_object",
    )
    assert tuple(value.value for value in engine.model_spec.reasoning_fields) == (
        "enabled",
        "effort",
    )
    assert tuple(value.value for value in engine.model_spec.reasoning_efforts) == (
        "high",
        "max",
    )
    assert tuple(value.value for value in engine.model_spec.sampling_fields) == (
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "logprobs",
        "top_logprobs",
    )


def test_unknown_model_is_rejected_before_client_use() -> None:
    with pytest.raises(UnsupportedModelError, match="future-model"):
        ChatCompletionsEngine(client=cast("Client", object()), model="future-model")


def test_execute_translates_normalized_request_and_response() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(
            200,
            headers={"x-request-id": "request-id", "retry-after": "1.25"},
            json=_chat_json(
                choices=[
                    _choice(
                        index=1, finish_reason="length", content="second", reasoning_content=None
                    ),
                    _choice(index=0),
                ]
            ),
        )

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        response = engine.execute(
            LanguageModelRequest(
                messages=(
                    SystemMessage(content=(TextContent(text="sys"), TextContent(text="tem"))),
                    UserMessage(
                        content=(TextContent(text="ques"), TextContent(text="tion")),
                    ),
                    AssistantMessage(content=(TextContent(text="prior"),)),
                ),
                response_format=JsonObjectResponseFormat(),
                reasoning=ReasoningConfig(enabled=True, effort=ReasoningEffort.MAX),
                sampling=SamplingConfig(
                    max_tokens=512,
                    stop=("END",),
                    logprobs=True,
                    top_logprobs=4,
                ),
                user="customer_42",
            )
        )
    finally:
        engine.close()

    assert captured_body == {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "prior"},
        ],
        "model": "deepseek-v4-flash",
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "stop": ["END"],
        "logprobs": True,
        "top_logprobs": 4,
        "user_id": "customer_42",
    }
    assert tuple(output.index for output in response.outputs) == (0, 1)
    assert response.outputs[0].text == "answer"
    assert response.outputs[0].message.reasoning == TextContent(text="thought")
    assert response.outputs[0].finish_reason is FinishReason.STOP
    assert response.outputs[1].text == "second"
    assert response.outputs[1].finish_reason is FinishReason.LENGTH
    assert response.metadata.provider == "deepseek"
    assert response.metadata.requested_model == "deepseek-v4-flash"
    assert response.metadata.response_model == "deepseek-v4-flash"
    assert response.metadata.request_id == "request-id"
    assert response.metadata.retry_after == 1.25
    assert response.metadata.response_id == "response-id"
    assert response.metadata.created_at == 10
    assert response.metadata.system_fingerprint == "fingerprint"
    assert response.metadata.rate_limit is None
    assert response.metadata.usage is not None
    assert response.metadata.usage.prompt_tokens == 11
    assert response.metadata.usage.completion_tokens == 7
    assert response.metadata.usage.total_tokens == 18
    assert response.metadata.usage.cached_prompt_tokens == 3
    assert response.metadata.usage.cache_miss_prompt_tokens == 8
    assert response.metadata.usage.reasoning_tokens == 2


def test_reasoning_disabled_maps_to_provider_thinking_control() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(200, json=_chat_json())

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        engine.execute(LanguageModelRequest(
            messages=(UserMessage(content=(TextContent(text="hello"),)),),
            reasoning=ReasoningConfig(enabled=False),
            sampling=SamplingConfig(temperature=0.5, top_p=0.9),
        ))
    finally:
        engine.close()

    assert captured_body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in captured_body
    assert captured_body["temperature"] == 0.5
    assert captured_body["top_p"] == 0.9
    assert "response_format" not in captured_body


def _unsupported_requests() -> list[LanguageModelRequest]:
    user = UserMessage(content=(TextContent(text="hello"),))
    return [
        LanguageModelRequest(messages=(DeveloperMessage(content=(TextContent(text="x"),)),)),
        LanguageModelRequest(
            messages=(UserMessage(content=(ImageContent(url="https://example.com/image.png"),)),)
        ),
        LanguageModelRequest(
            messages=(user,),
            response_format=JsonSchemaResponseFormat(
                name="answer",
                json_schema=JsonObject.parse({"type": "object"}),
                strict=True,
            ),
        ),
        LanguageModelRequest(
            messages=(
                AssistantMessage(
                    content=(TextContent(text="prefix"),),
                    reasoning=TextContent(text="prior thought"),
                ),
            )
        ),
        LanguageModelRequest(messages=(user,), reasoning=ReasoningConfig(clear=True)),
        LanguageModelRequest(
            messages=(user,),
            reasoning=ReasoningConfig(summary=ReasoningSummary.AUTO),
        ),
        LanguageModelRequest(
            messages=(user,),
            reasoning=ReasoningConfig(format=ReasoningFormat.PARSED),
        ),
        LanguageModelRequest(
            messages=(user,),
            reasoning=ReasoningConfig(effort=ReasoningEffort.MINIMAL),
        ),
        LanguageModelRequest(
            messages=(user,),
            reasoning=ReasoningConfig(enabled=False, effort=ReasoningEffort.HIGH),
        ),
        LanguageModelRequest(messages=(user,), sampling=SamplingConfig(max_tokens=384_001)),
        LanguageModelRequest(
            messages=(user,), sampling=SamplingConfig(stop=tuple(str(i) for i in range(17)))
        ),
        LanguageModelRequest(messages=(user,), sampling=SamplingConfig(seed=1)),
        LanguageModelRequest(messages=(user,), sampling=SamplingConfig(frequency_penalty=0.1)),
        LanguageModelRequest(messages=(user,), sampling=SamplingConfig(presence_penalty=0.1)),
        LanguageModelRequest(
            messages=(user,),
            sampling=SamplingConfig(logit_bias=(LogitBias(token="1", value=1),)),
        ),
        LanguageModelRequest(messages=(user,), sampling=SamplingConfig(top_logprobs=1)),
        LanguageModelRequest(messages=(user,), sampling=SamplingConfig(temperature=0.5)),
        LanguageModelRequest(
            messages=(user,),
            reasoning=ReasoningConfig(enabled=True),
            sampling=SamplingConfig(top_p=0.5),
        ),
        LanguageModelRequest(messages=(user,), metadata=(MetadataLabel(key="trace", value="x"),)),
        LanguageModelRequest(messages=(user,), user="not valid"),
    ]


@pytest.mark.parametrize("model_request", _unsupported_requests())
def test_unsupported_features_fail_before_client_invocation(
    model_request: LanguageModelRequest,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        msg = "client must not be called"
        raise AssertionError(msg)

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        with pytest.raises(UnsupportedFeatureError):
            engine.execute(model_request)
    finally:
        engine.close()

    assert calls == 0


def test_non_thinking_output_uses_none_reasoning() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(choices=[_choice(reasoning_content=None)]),
        )

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        response = engine.execute(LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),)))
    finally:
        engine.close()

    assert response.outputs[0].message.reasoning is None
    assert response.outputs[0].text == "answer"


def test_reasoning_only_output_is_preserved_without_invented_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(choices=[_choice(content=None, reasoning_content="thought")]),
        )

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        response = engine.execute(LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),)))
    finally:
        engine.close()

    assert response.outputs[0].message.content == ()
    assert response.outputs[0].message.reasoning == TextContent(text="thought")


def test_content_filtered_null_output_is_preserved_without_invented_text() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(
                choices=[
                    _choice(
                        finish_reason="content_filter",
                        content=None,
                        reasoning_content=None,
                    )
                ]
            ),
        )

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        response = engine.execute(LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),)))
    finally:
        engine.close()

    output = response.outputs[0]
    assert output.finish_reason is FinishReason.CONTENT_FILTER
    assert output.message.content == ()
    assert output.text == ""
    assert output.refusal is None


def test_optional_usage_counters_normalize_without_false_relations() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_cache_hit_tokens": 3,
                }
            ),
        )

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        response = engine.execute(LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),)))
    finally:
        engine.close()

    assert response.metadata.usage is not None
    assert response.metadata.usage.cached_prompt_tokens == 3
    assert response.metadata.usage.reasoning_tokens == 0
    assert response.metadata.usage.cache_miss_prompt_tokens == 0


@pytest.mark.parametrize(
    ("choice", "message"),
    [
        (_choice(index=-1), "index"),
        (_choice(finish_reason="tool_calls"), "finish reason"),
        (_choice(role="user"), "assistant"),
    ],
)
def test_malformed_choices_are_rejected(
    choice: dict[str, JsonValue],
    message: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_json(choices=[choice]))

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        with pytest.raises(InvalidResponseError, match=message):
            engine.execute(
                LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),))
            )
    finally:
        engine.close()


def test_duplicate_choice_indices_are_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_json(choices=[_choice(), _choice()]))

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        with pytest.raises(InvalidResponseError, match="duplicate"):
            engine.execute(LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),)))
    finally:
        engine.close()


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 99},
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
            "prompt_cache_hit_tokens": 2,
            "prompt_cache_miss_tokens": 2,
        },
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
        {"prompt_tokens": -1, "completion_tokens": 3, "total_tokens": 2},
    ],
)
def test_inconsistent_or_negative_usage_is_rejected(usage: dict[str, JsonValue]) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_json(usage=usage))

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        with pytest.raises(InvalidResponseError, match="usage"):
            engine.execute(LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),)))
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [
        (ReasoningEffort.HIGH, "high"),
        (ReasoningEffort.MAX, "max"),
    ],
)
def test_supported_reasoning_efforts_are_transported_exactly(
    effort: ReasoningEffort,
    wire_effort: str,
) -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(200, json=_chat_json())

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        engine.execute(LanguageModelRequest(
            messages=(UserMessage(content=(TextContent(text="hello"),)),),
            reasoning=ReasoningConfig(effort=effort),
        ))
    finally:
        engine.close()

    assert captured_body["reasoning_effort"] == wire_effort


@pytest.mark.parametrize(
    "effort",
    [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.XHIGH],
)
def test_unsupported_reasoning_efforts_fail_before_transport(
    effort: ReasoningEffort,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        msg = "client must not be called"
        raise AssertionError(msg)

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        with pytest.raises(UnsupportedFeatureError, match=effort.value):
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                    reasoning=ReasoningConfig(effort=effort),
                )
            )
    finally:
        engine.close()

    assert calls == 0


def test_response_model_identity_is_preserved_without_exact_equality_rejection() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(model="provider-resolved-deepseek-model-2026-07-01"),
        )

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        response = engine.execute(LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),)))
    finally:
        engine.close()

    assert response.outputs[0].text == "answer"
    assert response.metadata.requested_model == "deepseek-v4-flash"
    assert response.metadata.response_model == "provider-resolved-deepseek-model-2026-07-01"


def test_invalid_response_object_is_rejected_by_wire_parser() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_json(object_type="future.object"))

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        with pytest.raises(InvalidResponseError) as caught:
            engine.execute(LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),)))
    finally:
        engine.close()

    assert isinstance(caught.value.__cause__, deepseek_errors.ResponseError)


def test_empty_choices_are_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_json(choices=[]))

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        with pytest.raises(InvalidResponseError, match="choices"):
            engine.execute(LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),)))
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("provider_error", "runtime_error"),
    [
        (
            deepseek_errors.AuthError(
                DeepSeekResponseMetadata(status_code=401, request_id="auth-id", retry_after=None),
                "secret body",
            ),
            AuthenticationError,
        ),
        (
            deepseek_errors.RateLimitError(
                DeepSeekResponseMetadata(status_code=429, request_id="rate-id", retry_after=2.5),
                "secret body",
            ),
            RateLimitError,
        ),
        (
            deepseek_errors.APIError(
                DeepSeekResponseMetadata(status_code=500, request_id="api-id", retry_after=None),
                "secret body",
            ),
            ExecutionError,
        ),
        (
            deepseek_errors.ResponseError(
                "invalid body",
                metadata=DeepSeekResponseMetadata(
                    status_code=200,
                    request_id="response-id",
                    retry_after=None,
                ),
                body="secret body",
            ),
            InvalidResponseError,
        ),
        (deepseek_errors.TransportError("offline"), TransportError),
    ],
)
def test_provider_errors_are_normalized_with_chained_cause(
    provider_error: Exception,
    runtime_error: type[Exception],
) -> None:
    class FailingClient:
        def create_chat_completion(self, _request: object) -> object:
            raise provider_error

    engine = ChatCompletionsEngine(
        client=cast("Client", FailingClient()),
        model="deepseek-v4-flash",
    )
    with pytest.raises(runtime_error) as caught:
        engine.execute(
            LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),))
        )

    assert caught.value.__cause__ is provider_error
    assert "secret body" not in str(caught.value)


def test_invalid_json_retains_decode_evidence() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        with pytest.raises(InvalidResponseError) as caught:
            engine.execute(
                LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),))
            )
    finally:
        engine.close()

    assert isinstance(caught.value.__cause__, deepseek_errors.ResponseError)
    assert isinstance(caught.value.__cause__.__cause__, json.JSONDecodeError)


def test_invalid_schema_retains_validation_evidence() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"future": "shape"})

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="deepseek-v4-flash")
    try:
        with pytest.raises(InvalidResponseError) as caught:
            engine.execute(
                LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),))
            )
    finally:
        engine.close()

    assert isinstance(caught.value.__cause__, deepseek_errors.ResponseError)
    assert isinstance(caught.value.__cause__.__cause__, ValidationError)
