import json
from collections.abc import Callable
from typing import cast

import httpx
import pytest
from pydantic import JsonValue, ValidationError

from symai.backend.engines.language_model.cerebras import LanguageModelEngine
from symai.clients.cerebras import errors as cerebras_errors
from symai.clients.cerebras.chat import CreateChatCompletionRequest
from symai.clients.cerebras.client import Client as CerebrasClient
from symai.clients.cerebras.transport import RateLimitState
from symai.clients.cerebras.transport import ResponseMetadata as CerebrasResponseMetadata
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
    ImageDetail,
    JsonObject,
    JsonSchemaResponseFormat,
    LanguageModelRequest,
    LogitBias,
    MetadataLabel,
    Provider,
    ReasoningConfig,
    ReasoningEffort,
    ReasoningFormat,
    ReasoningSummary,
    SamplingConfig,
    SystemMessage,
    TextContent,
    UserMessage,
)

_RATE_LIMIT_HEADERS = {
    "x-ratelimit-limit-requests-day": "100",
    "x-ratelimit-limit-tokens-minute": "1000",
    "x-ratelimit-remaining-requests-day": "99",
    "x-ratelimit-remaining-tokens-minute": "900",
    "x-ratelimit-reset-requests-day": "30.5",
    "x-ratelimit-reset-tokens-minute": "5.5",
}


def _choice(
    *,
    index: int | None = 0,
    finish_reason: str | None = "stop",
    content: str | None = "answer",
    reasoning: str | None = "thought",
    role: str | None = "assistant",
) -> dict[str, JsonValue]:
    return {
        "finish_reason": finish_reason,
        "index": index,
        "message": {
            "role": role,
            "content": content,
            "reasoning": reasoning,
        },
    }


def _chat_json(
    *,
    choices: list[dict[str, JsonValue]] | None = None,
    usage: dict[str, JsonValue] | None = None,
    model: str = "gpt-oss-120b",
) -> dict[str, object]:
    return {
        "id": "response-id",
        "choices": choices if choices is not None else [_choice()],
        "created": 10,
        "model": model,
        "object": "chat.completion",
        "system_fingerprint": "fingerprint",
        "usage": usage
        if usage is not None
        else {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "image_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {
                "reasoning_tokens": 2,
                "accepted_prediction_tokens": 1,
                "rejected_prediction_tokens": 1,
            },
        },
    }


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[CerebrasClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return CerebrasClient(api_key="test-key", http_client=http_client), http_client


def test_execute_translates_normalized_request_and_response() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(
            200,
            headers={
                "x-request-id": "request-id",
                "retry-after": "1.25",
                **_RATE_LIMIT_HEADERS,
            },
            json=_chat_json(
                choices=[
                    _choice(index=1, finish_reason="length", content="second", reasoning=None),
                    _choice(index=0),
                ]
            ),
        )

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-oss-120b")
        response = engine.execute(
            LanguageModelRequest(
                messages=(
                    SystemMessage(content=(TextContent(text="system"),)),
                    DeveloperMessage(content=(TextContent(text="developer"),)),
                    UserMessage(
                        content=(
                            TextContent(text="question"),
                            ImageContent(url="https://example.com/image.png"),
                        )
                    ),
                    AssistantMessage(
                        content=(TextContent(text="prior"),),
                        reasoning=TextContent(text="prior thought"),
                    ),
                ),
                response_format=JsonSchemaResponseFormat(
                    name="answer",
                    json_schema=JsonObject.parse(
                        {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        }
                    ),
                    description="Answer shape",
                    strict=True,
                ),
                reasoning=ReasoningConfig(
                    effort=ReasoningEffort.HIGH,
                    format=ReasoningFormat.PARSED,
                    clear=False,
                ),
                sampling=SamplingConfig(
                    max_tokens=512,
                    temperature=0.5,
                    top_p=0.9,
                    stop=("END",),
                    seed=42,
                    frequency_penalty=-0.5,
                    logit_bias=(
                        LogitBias(token="123", value=-100),
                        LogitBias(token="456", value=100),
                    ),
                    presence_penalty=0.5,
                    logprobs=True,
                    top_logprobs=4,
                ),
                user="customer-42",
            )
        )
        assert http_client.is_closed is False
    finally:
        http_client.close()

    assert captured_body == {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "developer", "content": [{"type": "text", "text": "developer"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "question"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "prior"}],
                "reasoning": "prior thought",
            },
        ],
        "model": "gpt-oss-120b",
        "clear_thinking": False,
        "frequency_penalty": -0.5,
        "logit_bias": {"123": -100.0, "456": 100.0},
        "logprobs": True,
        "max_completion_tokens": 512,
        "presence_penalty": 0.5,
        "reasoning_effort": "high",
        "reasoning_format": "parsed",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "description": "Answer shape",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                "strict": True,
            },
        },
        "seed": 42,
        "stop": ["END"],
        "temperature": 0.5,
        "top_logprobs": 4,
        "top_p": 0.9,
        "user": "customer-42",
    }
    assert tuple(output.index for output in response.outputs) == (0, 1)
    assert response.outputs[0].text == "answer"
    assert response.outputs[0].message.reasoning == TextContent(text="thought")
    assert response.outputs[0].finish_reason is FinishReason.STOP
    assert response.outputs[1].text == "second"
    assert response.outputs[1].finish_reason is FinishReason.LENGTH
    assert response.metadata.provider is Provider.CEREBRAS
    assert response.metadata.requested_model == "gpt-oss-120b"
    assert response.metadata.response_model == "gpt-oss-120b"
    assert response.metadata.request_id == "request-id"
    assert response.metadata.retry_after == 1.25
    assert response.metadata.response_id == "response-id"
    assert response.metadata.created_at == 10
    assert response.metadata.system_fingerprint == "fingerprint"
    assert response.metadata.usage is not None
    assert response.metadata.usage.prompt_tokens == 11
    assert response.metadata.usage.completion_tokens == 7
    assert response.metadata.usage.total_tokens == 18
    assert response.metadata.usage.cached_prompt_tokens == 3
    assert response.metadata.usage.reasoning_tokens == 2
    assert response.metadata.usage.image_tokens == 2
    assert response.metadata.usage.accepted_prediction_tokens == 1
    assert response.metadata.usage.rejected_prediction_tokens == 1
    assert response.metadata.rate_limit is not None
    assert response.metadata.rate_limit.limit_requests_day == 100
    assert response.metadata.rate_limit.remaining_tokens_minute == 900
    assert response.metadata.rate_limit.reset_tokens_minute == 5.5


def test_response_model_identity_is_preserved_without_prefix_matching() -> None:
    returned_model = "provider-resolved-cerebras-model-2026-07-15"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_json(model=returned_model))

    client, http_client = _client(handler)
    try:
        response = LanguageModelEngine(client=client, model="gpt-oss-120b").execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        http_client.close()

    assert response.metadata.requested_model == "gpt-oss-120b"
    assert response.metadata.response_model == returned_model


def test_non_thinking_effort_sentinel_is_model_specific() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(
            200,
            json=_chat_json(model="zai-glm-4.7", choices=[_choice(reasoning=None)]),
        )

    client, http_client = _client(handler)
    try:
        response = LanguageModelEngine(client=client, model="zai-glm-4.7").execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                reasoning=ReasoningConfig(effort=ReasoningEffort.NONE),
            )
        )
    finally:
        http_client.close()

    assert captured_body["reasoning_effort"] == "none"
    assert response.outputs[0].message.reasoning is None


@pytest.mark.parametrize(
    ("model", "model_request"),
    [
        (
            "gpt-oss-120b",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                reasoning=ReasoningConfig(effort=ReasoningEffort.NONE),
            ),
        ),
        (
            "gpt-oss-120b",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                reasoning=ReasoningConfig(enabled=False),
            ),
        ),
        (
            "gpt-oss-120b",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                reasoning=ReasoningConfig(summary=ReasoningSummary.AUTO),
            ),
        ),
        (
            "gpt-oss-120b",
            LanguageModelRequest(
                messages=(
                    UserMessage(
                        content=(
                            ImageContent(
                                url="https://example.com/image.png",
                                detail=ImageDetail.HIGH,
                            ),
                        )
                    ),
                ),
            ),
        ),
        (
            "gpt-oss-120b",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                sampling=SamplingConfig(max_tokens=40_001),
            ),
        ),
        (
            "gpt-oss-120b",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                sampling=SamplingConfig(stop=("1", "2", "3", "4", "5")),
            ),
        ),
        (
            "gpt-oss-120b",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                metadata=(MetadataLabel(key="trace", value="abc"),),
            ),
        ),
    ],
)
def test_unsupported_features_fail_before_client_invocation(
    model: str,
    model_request: LanguageModelRequest,
) -> None:
    calls = 0

    class RecordingClient:
        def create_chat_completion(self, _request: CreateChatCompletionRequest):
            nonlocal calls
            calls += 1
            msg = "client must not be called"
            raise AssertionError(msg)

    engine = LanguageModelEngine(client=cast("CerebrasClient", RecordingClient()), model=model)

    with pytest.raises(UnsupportedFeatureError):
        engine.execute(model_request)

    assert calls == 0


def test_unknown_model_is_rejected_at_construction() -> None:
    with pytest.raises(UnsupportedModelError, match="future-model"):
        LanguageModelEngine(
            client=cast("CerebrasClient", object()),
            model="future-model",
        )


def test_optional_usage_counters_normalize_without_invented_relations() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(usage={"prompt_tokens": 10}),
        )

    client, http_client = _client(handler)
    try:
        response = LanguageModelEngine(client=client, model="gpt-oss-120b").execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        http_client.close()

    assert response.metadata.usage is not None
    assert response.metadata.usage.prompt_tokens == 10
    assert response.metadata.usage.completion_tokens == 0
    assert response.metadata.usage.total_tokens == 0
    assert response.metadata.rate_limit is None


@pytest.mark.parametrize(
    ("choice", "message"),
    [
        (_choice(index=None), "index"),
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

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-oss-120b")
        with pytest.raises(InvalidResponseError, match=message):
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        http_client.close()


def test_duplicate_choice_indices_are_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(choices=[_choice(index=0), _choice(index=0)]),
        )

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-oss-120b")
        with pytest.raises(InvalidResponseError, match="duplicate"):
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        http_client.close()


def test_inconsistent_reported_usage_is_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99}
            ),
        )

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-oss-120b")
        with pytest.raises(InvalidResponseError, match="token usage"):
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        http_client.close()


@pytest.mark.parametrize(
    ("provider_error", "runtime_error"),
    [
        (
            cerebras_errors.AuthError(
                CerebrasResponseMetadata(
                    status_code=401,
                    request_id="auth-id",
                    retry_after=None,
                    rate_limit=RateLimitState(),
                ),
                "secret body",
            ),
            AuthenticationError,
        ),
        (
            cerebras_errors.RateLimitError(
                CerebrasResponseMetadata(
                    status_code=429,
                    request_id="rate-id",
                    retry_after=2.0,
                    rate_limit=RateLimitState(),
                ),
                "secret body",
            ),
            RateLimitError,
        ),
        (cerebras_errors.TransportError("network failed"), TransportError),
        (
            cerebras_errors.ResponseError(
                "invalid response",
                metadata=CerebrasResponseMetadata(
                    status_code=200,
                    request_id="response-id",
                    retry_after=None,
                    rate_limit=RateLimitState(),
                ),
                body="secret body",
            ),
            InvalidResponseError,
        ),
        (
            cerebras_errors.APIError(
                CerebrasResponseMetadata(
                    status_code=500,
                    request_id="api-id",
                    retry_after=None,
                    rate_limit=RateLimitState(),
                ),
                "secret body",
            ),
            ExecutionError,
        ),
    ],
)
def test_provider_errors_are_normalized_with_chained_cause(
    provider_error: Exception,
    runtime_error: type[ExecutionError],
) -> None:
    class FailingClient:
        def create_chat_completion(self, _request: CreateChatCompletionRequest):
            raise provider_error

    engine = LanguageModelEngine(
        client=cast("CerebrasClient", FailingClient()),
        model="gpt-oss-120b",
    )

    with pytest.raises(runtime_error) as caught:
        engine.execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )

    assert caught.value.__cause__ is provider_error
    assert "secret body" not in str(caught.value)
    if isinstance(
        provider_error,
        (cerebras_errors.APIError, cerebras_errors.ResponseError),
    ):
        assert caught.value.metadata is not None
        assert caught.value.metadata.request_id == provider_error.metadata.request_id
        assert caught.value.metadata.retry_after == provider_error.metadata.retry_after


def test_invalid_normalized_response_is_chained() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(choices=[_choice(content=None, reasoning=None)]),
        )

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-oss-120b")
        with pytest.raises(InvalidResponseError) as caught:
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        http_client.close()

    assert isinstance(caught.value.__cause__, ValidationError)
