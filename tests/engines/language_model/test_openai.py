import json
from collections.abc import Callable
from typing import cast

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines.language_model.openai import LanguageModelEngine
from symai.clients.openai import errors as openai_errors
from symai.clients.openai.client import Client as OpenAIClient
from symai.clients.openai.responses import CreateResponseRequest
from symai.clients.openai.transport import ResponseMetadata as OpenAIResponseMetadata
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


def _response_json(
    *,
    status: str = "completed",
    model: str = "gpt-5.4",
    output: list[dict[str, object]] | None = None,
    reasoning: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    response_output = list(reasoning or ())
    response_output.extend(
        output
        if output is not None
        else [
            {
                "id": "message-id",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "answer",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ]
    )
    return {
        "id": "response-id",
        "object": "response",
        "created_at": 1.5,
        "status": status,
        "background": status != "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 512,
        "model": model,
        "output": response_output,
        "store": False,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 11,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 18,
        },
        "metadata": {},
    }


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[OpenAIClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIClient(api_key="test-key", http_client=http_client), http_client


def test_execute_translates_normalized_request_and_response() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(
            200,
            headers={"x-request-id": "request-id", "retry-after": "1.25"},
            json=_response_json(
                reasoning=[
                    {
                        "id": "reasoning-id",
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [{"type": "summary_text", "text": "thought"}],
                    }
                ]
            ),
        )

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-5.4")
        request = LanguageModelRequest(
            messages=(
                SystemMessage(content=(TextContent(text="system"),)),
                DeveloperMessage(content=(TextContent(text="developer"),)),
                UserMessage(
                    content=(
                        TextContent(text="question"),
                        ImageContent(url="https://example.com/image.png", detail=ImageDetail.HIGH),
                    )
                ),
                AssistantMessage(content=(TextContent(text="prior"),)),
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
                summary=ReasoningSummary.DETAILED,
            ),
            sampling=SamplingConfig(max_tokens=512, top_logprobs=4),
            user="customer-42",
            metadata=(MetadataLabel(key="trace", value="abc"),),
        )

        response = engine.execute(request)
    finally:
        http_client.close()

    assert captured_body == {
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "system"}],
                "type": "message",
            },
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer"}],
                "type": "message",
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "question"},
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": "https://example.com/image.png",
                    },
                ],
                "type": "message",
            },
            {
                "role": "assistant",
                "content": [{"type": "input_text", "text": "prior"}],
                "type": "message",
            },
        ],
        "model": "gpt-5.4",
        "max_output_tokens": 512,
        "metadata": {"trace": "abc"},
        "reasoning": {"effort": "high", "summary": "detailed"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "description": "Answer shape",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                "strict": True,
            }
        },
        "top_logprobs": 4,
        "user": "customer-42",
    }
    assert len(response.outputs) == 1
    assert response.outputs[0].text == "answer"
    assert response.outputs[0].message.reasoning == TextContent(text="thought")
    assert response.metadata.provider.value == "openai"
    assert response.metadata.model == "gpt-5.4"
    assert response.metadata.status_code == 200
    assert response.metadata.request_id == "request-id"
    assert response.metadata.retry_after == 1.25
    assert response.metadata.response_id == "response-id"
    assert response.metadata.created_at == 1.5
    assert response.metadata.usage is not None
    assert response.metadata.usage.prompt_tokens == 11
    assert response.metadata.usage.completion_tokens == 7
    assert response.metadata.usage.cached_prompt_tokens == 3
    assert response.metadata.usage.reasoning_tokens == 2


def test_nonreasoning_model_maps_supported_sampling_and_default_text_format() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(200, json=_response_json(model="gpt-4.1"))

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-4.1")
        engine.execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                sampling=SamplingConfig(
                    max_tokens=128,
                    temperature=0.25,
                    top_p=0.8,
                    top_logprobs=2,
                ),
            )
        )
    finally:
        http_client.close()

    assert captured_body["temperature"] == 0.25
    assert captured_body["top_p"] == 0.8
    assert captured_body["top_logprobs"] == 2
    assert captured_body["text"] == {"format": {"type": "text"}}
    assert "reasoning" not in captured_body


def test_reasoning_model_uses_explicit_default_effort() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(200, json=_response_json(model="gpt-5.4-pro"))

    client, http_client = _client(handler)
    try:
        LanguageModelEngine(client=client, model="gpt-5.4-pro").execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        http_client.close()

    assert captured_body["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize(
    ("model", "model_request"),
    [
        (
            "gpt-4.1",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                reasoning=ReasoningConfig(effort=ReasoningEffort.MEDIUM),
            ),
        ),
        (
            "gpt-5.4",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                sampling=SamplingConfig(temperature=0.5),
            ),
        ),
        (
            "gpt-5.4",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                sampling=SamplingConfig(stop=("END",)),
            ),
        ),
        (
            "gpt-5.4",
            LanguageModelRequest(
                messages=(AssistantMessage(reasoning=TextContent(text="private reasoning")),),
            ),
        ),
        (
            "gpt-5.4",
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
                reasoning=ReasoningConfig(format=ReasoningFormat.HIDDEN),
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
        def create_response(self, _request: CreateResponseRequest):
            nonlocal calls
            calls += 1
            msg = "client must not be called"
            raise AssertionError(msg)

    engine = LanguageModelEngine(client=cast("OpenAIClient", RecordingClient()), model=model)

    with pytest.raises(UnsupportedFeatureError):
        engine.execute(model_request)

    assert calls == 0


def test_unknown_model_is_rejected_at_construction() -> None:
    class RecordingClient:
        def create_response(self, _request: CreateResponseRequest):
            msg = "client must not be called"
            raise AssertionError(msg)

    with pytest.raises(UnsupportedModelError, match="future-model"):
        LanguageModelEngine(
            client=cast("OpenAIClient", RecordingClient()),
            model="future-model",
        )


@pytest.mark.parametrize(
    ("provider_error", "runtime_error"),
    [
        (
            openai_errors.AuthError(
                OpenAIResponseMetadata(status_code=401, request_id="auth-id", retry_after=None),
                "secret body",
            ),
            AuthenticationError,
        ),
        (
            openai_errors.RateLimitError(
                OpenAIResponseMetadata(status_code=429, request_id="rate-id", retry_after=2.0),
                "secret body",
            ),
            RateLimitError,
        ),
        (
            openai_errors.TransportError("network failed"),
            TransportError,
        ),
        (
            openai_errors.ResponseError(
                "invalid response",
                metadata=OpenAIResponseMetadata(
                    status_code=200,
                    request_id="response-id",
                    retry_after=None,
                ),
                body="secret body",
            ),
            InvalidResponseError,
        ),
        (
            openai_errors.APIError(
                OpenAIResponseMetadata(status_code=500, request_id="api-id", retry_after=None),
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
        def create_response(self, _request: CreateResponseRequest):
            raise provider_error

    engine = LanguageModelEngine(client=cast("OpenAIClient", FailingClient()), model="gpt-5.4")
    request = LanguageModelRequest(
        messages=(UserMessage(content=(TextContent(text="hello"),)),),
    )

    with pytest.raises(runtime_error) as caught:
        engine.execute(request)

    assert caught.value.__cause__ is provider_error
    assert "secret body" not in str(caught.value)
    if isinstance(provider_error, (openai_errors.APIError, openai_errors.ResponseError)):
        assert caught.value.metadata is not None
        assert caught.value.metadata.request_id == provider_error.metadata.request_id
        assert caught.value.metadata.retry_after == provider_error.metadata.retry_after


def test_dated_response_model_preserves_requested_and_returned_identity() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response_json(model="gpt-5.4-2026-03-05"),
        )

    client, http_client = _client(handler)
    try:
        response = LanguageModelEngine(client=client, model="gpt-5.4").execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        http_client.close()

    assert response.outputs[0].text == "answer"
    assert response.metadata.requested_model == "gpt-5.4"
    assert response.metadata.response_model == "gpt-5.4-2026-03-05"


def test_refusal_only_output_is_normalized_without_invented_text() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response_json(
                output=[
                    {
                        "id": "message-id",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "refusal", "refusal": "cannot comply"}],
                    }
                ]
            ),
        )

    client, http_client = _client(handler)
    try:
        response = LanguageModelEngine(client=client, model="gpt-5.4").execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        http_client.close()

    assert response.outputs[0].text == ""
    assert response.outputs[0].refusal == "cannot comply"


@pytest.mark.parametrize("status", ["queued", "in_progress", "cancelled"])
def test_noncompleted_response_is_invalid(status: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _response_json(status=status)
        if status == "failed":
            payload["error"] = {"code": "server_error", "message": "provider failed"}
        return httpx.Response(200, json=payload)

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-5.4")
        with pytest.raises(InvalidResponseError, match=status):
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        http_client.close()


def test_noncompleted_response_does_not_expose_provider_error_text() -> None:
    provider_message = "distinctive provider-controlled secret"

    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _response_json(status="queued")
        payload["error"] = {"code": "server_error", "message": provider_message}
        return httpx.Response(200, json=payload)

    client, http_client = _client(handler)
    try:
        with pytest.raises(InvalidResponseError) as caught:
            LanguageModelEngine(client=client, model="gpt-5.4").execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        http_client.close()

    assert provider_message not in str(caught.value)
    assert caught.value.metadata is not None
    assert caught.value.metadata.model == "gpt-5.4"


def test_failed_response_with_usable_output_maps_error_finish_reason() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _response_json(status="failed")
        payload["error"] = {"code": "server_error", "message": "provider failed"}
        return httpx.Response(200, json=payload)

    client, http_client = _client(handler)
    try:
        response = LanguageModelEngine(client=client, model="gpt-5.4").execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        http_client.close()

    assert response.outputs[0].text == "answer"
    assert response.outputs[0].finish_reason is FinishReason.ERROR


def test_inconsistent_reported_usage_is_invalid() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _response_json()
        usage = payload["usage"]
        assert isinstance(usage, dict)
        usage["total_tokens"] = 99
        return httpx.Response(200, json=payload)

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-5.4")
        with pytest.raises(InvalidResponseError, match="token usage"):
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        http_client.close()


@pytest.mark.parametrize(
    ("reason", "finish_reason"),
    [
        ("max_output_tokens", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
    ],
)
def test_incomplete_terminal_response_preserves_partial_output(
    reason: str,
    finish_reason: FinishReason,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _response_json(
            status="incomplete",
            output=[
                {
                    "id": "message-id",
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "partial",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            ],
        )
        payload["incomplete_details"] = {"reason": reason}
        return httpx.Response(200, json=payload)

    client, http_client = _client(handler)
    try:
        response = LanguageModelEngine(client=client, model="gpt-5.4").execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="hello"),)),),
            )
        )
    finally:
        http_client.close()

    assert response.outputs[0].text == "partial"
    assert response.outputs[0].finish_reason is finish_reason


def test_invalid_provider_metadata_is_normalized() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _response_json()
        payload["created_at"] = -1
        return httpx.Response(200, json=payload)

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-5.4")
        with pytest.raises(InvalidResponseError, match="metadata") as caught:
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        http_client.close()

    assert isinstance(caught.value.__cause__, ValidationError)


def test_nonfinite_provider_retry_after_does_not_mask_rate_limit_error() -> None:
    provider_error = openai_errors.RateLimitError(
        OpenAIResponseMetadata(status_code=429, request_id="rate-id", retry_after=float("inf")),
        "secret body",
    )

    class FailingClient:
        def create_response(self, _request: CreateResponseRequest):
            raise provider_error

    engine = LanguageModelEngine(client=cast("OpenAIClient", FailingClient()), model="gpt-5.4")
    request = LanguageModelRequest(
        messages=(UserMessage(content=(TextContent(text="hello"),)),),
    )

    with pytest.raises(RateLimitError) as caught:
        engine.execute(request)

    assert caught.value.__cause__ is provider_error
    assert caught.value.metadata is not None
    assert caught.value.metadata.retry_after is None


def test_unsupported_provider_output_item_is_not_silently_dropped() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _response_json()
        output = payload["output"]
        assert isinstance(output, list)
        output.insert(
            0,
            {
                "id": "compaction-id",
                "type": "compaction",
                "encrypted_content": "opaque",
            },
        )
        return httpx.Response(200, json=payload)

    client, http_client = _client(handler)
    try:
        engine = LanguageModelEngine(client=client, model="gpt-5.4")
        with pytest.raises(InvalidResponseError, match="unsupported output"):
            engine.execute(
                LanguageModelRequest(
                    messages=(UserMessage(content=(TextContent(text="hello"),)),),
                )
            )
    finally:
        http_client.close()
