import json
from collections.abc import Callable
from typing import cast

import httpx
import pytest
from pydantic import JsonValue, SecretStr, ValidationError

from symai.providers.cerebras.client import errors as cerebras_errors
from symai.providers.cerebras.client.chat import CreateChatCompletionRequest
from symai.providers.cerebras.client.client import Client
from symai.providers.cerebras.client.transport import RateLimitState
from symai.providers.cerebras.client.transport import ResponseMetadata as CerebrasResponseMetadata
from symai.providers.cerebras.engines.chat_completions import ChatCompletionsEngine
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
    JsonSchemaResponseFormat,
    LanguageModelRequest,
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
) -> Client:
    return Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )


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
                model="gemma-4-31b",
                choices=[
                    _choice(index=1, finish_reason="length", content="second", reasoning=None),
                    _choice(index=0),
                ],
            ),
        )

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="gemma-4-31b")
    try:
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
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    description="Answer shape",
                    strict=True,
                ),
                reasoning=ReasoningConfig(
                    effort=ReasoningEffort.HIGH,
                    format=ReasoningFormat.PARSED,
                ),
                sampling=SamplingConfig(
                    max_tokens=512,
                    temperature=0.5,
                    top_p=0.9,
                    stop=("END",),
                    seed=42,
                    frequency_penalty=-0.5,
                    presence_penalty=0.5,
                ),
                user="customer-42",
            )
        )
    finally:
        engine.close()

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
        "model": "gemma-4-31b",
        "frequency_penalty": -0.5,
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
        "top_p": 0.9,
        "user": "customer-42",
    }
    assert tuple(output.index for output in response.outputs) == (0, 1)
    assert response.outputs[0].text == "answer"
    assert response.outputs[0].message.reasoning == TextContent(text="thought")
    assert response.outputs[0].finish_reason is FinishReason.STOP
    assert response.outputs[1].text == "second"
    assert response.outputs[1].finish_reason is FinishReason.LENGTH
    assert response.metadata.provider == "cerebras"
    assert response.metadata.requested_model == "gemma-4-31b"
    assert response.metadata.response_model == "gemma-4-31b"
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

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="gpt-oss-120b")
    try:
        response = engine.execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        engine.close()

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

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="zai-glm-4.7")
    try:
        response = engine.execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                reasoning=ReasoningConfig(effort=ReasoningEffort.NONE),
            )
        )
    finally:
        engine.close()

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

    engine = ChatCompletionsEngine(client=cast("Client", RecordingClient()), model=model)

    with pytest.raises(UnsupportedFeatureError):
        engine.execute(model_request)

    assert calls == 0


def test_unknown_model_is_rejected_at_construction() -> None:
    with pytest.raises(UnsupportedModelError, match="future-model"):
        ChatCompletionsEngine(
            client=cast("Client", object()),
            model="future-model",
        )


def test_optional_usage_counters_normalize_without_invented_relations() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(usage={"prompt_tokens": 10}),
        )

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="gpt-oss-120b")
    try:
        response = engine.execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        engine.close()

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

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="gpt-oss-120b")
    try:
        with pytest.raises(InvalidResponseError, match=message):
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        engine.close()


def test_duplicate_choice_indices_are_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_json(choices=[_choice(index=0), _choice(index=0)]),
        )

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="gpt-oss-120b")
    try:
        with pytest.raises(InvalidResponseError, match="duplicate"):
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        engine.close()


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99},
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 11},
        },
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 6},
        },
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 3,
                "rejected_prediction_tokens": 3,
            },
        },
        {"prompt_tokens": -1, "completion_tokens": 5, "total_tokens": 4},
    ],
)
def test_inconsistent_reported_usage_is_omitted(usage: dict[str, JsonValue]) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_json(usage=usage))

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="gpt-oss-120b")
    try:
        response = engine.execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        engine.close()

    assert response.outputs[0].text == "answer"
    assert response.metadata.usage is None


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

    engine = ChatCompletionsEngine(
        client=cast("Client", FailingClient()),
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

    client = _client(handler)
    engine = ChatCompletionsEngine(client=client, model="gpt-oss-120b")
    try:
        with pytest.raises(InvalidResponseError) as caught:
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        engine.close()

    assert isinstance(caught.value.__cause__, ValidationError)


def _matrix_engine(model: str, captured: dict[str, object]) -> ChatCompletionsEngine:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.read()))
        return httpx.Response(200, json=_chat_json(model=model))

    return ChatCompletionsEngine(client=_client(handler), model=model)


def _vision_request() -> LanguageModelRequest:
    # No `detail`: Cerebras rejects normalized image detail on every model, which would
    # mask the per-model vision decision under test.
    return LanguageModelRequest(
        messages=(UserMessage(content=(ImageContent(url="https://example.com/i.png"),)),)
    )


def _reasoning_request(**kwargs: object) -> LanguageModelRequest:
    return LanguageModelRequest(
        messages=(UserMessage(content=(TextContent(text="hello"),)),),
        reasoning=ReasoningConfig(**kwargs),  # pyright: ignore[reportArgumentType]
    )


# Each capability is per-model: a model that lacks it must reject before any HTTP call,
# and the one that has it must still reach the wire.
@pytest.mark.parametrize(
    ("model", "supported"),
    [("gemma-4-31b", True), ("gpt-oss-120b", False), ("zai-glm-4.7", False)],
)
def test_vision_support_follows_the_model_spec(model: str, supported: bool) -> None:
    captured: dict[str, object] = {}
    engine = _matrix_engine(model, captured)
    try:
        if not supported:
            with pytest.raises(UnsupportedFeatureError):
                engine.execute(_vision_request())
            assert captured == {}
            return

        engine.execute(_vision_request())
        assert captured["model"] == model
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("model", "supported"),
    [("zai-glm-4.7", True), ("gpt-oss-120b", False), ("gemma-4-31b", False)],
)
def test_clear_thinking_support_follows_the_model_spec(model: str, supported: bool) -> None:
    captured: dict[str, object] = {}
    engine = _matrix_engine(model, captured)
    try:
        request = _reasoning_request(clear=True)
        if not supported:
            with pytest.raises(UnsupportedFeatureError):
                engine.execute(request)
            assert captured == {}
            return

        engine.execute(request)
        assert captured["clear_thinking"] is True
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("model", "supported"),
    [("gpt-oss-120b", True), ("zai-glm-4.7", True), ("gemma-4-31b", False)],
)
def test_raw_reasoning_format_support_follows_the_model_spec(model: str, supported: bool) -> None:
    captured: dict[str, object] = {}
    engine = _matrix_engine(model, captured)
    try:
        request = _reasoning_request(format=ReasoningFormat.RAW)
        if not supported:
            with pytest.raises(UnsupportedFeatureError):
                engine.execute(request)
            assert captured == {}
            return

        engine.execute(request)
        assert captured["reasoning_format"] == "raw"
    finally:
        engine.close()


def test_parsed_reasoning_format_is_supported_by_every_reasoning_model() -> None:
    for model in ("gpt-oss-120b", "gemma-4-31b", "zai-glm-4.7"):
        captured: dict[str, object] = {}
        engine = _matrix_engine(model, captured)
        try:
            engine.execute(_reasoning_request(format=ReasoningFormat.PARSED))
            assert captured["reasoning_format"] == "parsed"
        finally:
            engine.close()


def test_request_model_rejects_unknown_fields_like_the_other_providers() -> None:
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "gpt-oss-120b",
                "temperatur": 0.5,
            }
        )
