import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import ValidationError

from symai.clients.cerebras.chat import (
    ChatCompletion,
    CreateChatCompletionRequest,
    ReasoningFormat,
    UserMessage,
)
from symai.clients.cerebras.client import Client
from symai.clients.cerebras.errors import (
    APIError,
    AuthError,
    RateLimitError,
    ResponseError,
    TransportError,
)
from symai.clients.cerebras.transport import APIResponse

RATE_LIMIT_HEADERS = {
    "x-ratelimit-limit-requests-day": "100",
    "x-ratelimit-limit-tokens-minute": "1000",
    "x-ratelimit-remaining-requests-day": "99",
    "x-ratelimit-remaining-tokens-minute": "900",
    "x-ratelimit-reset-requests-day": "30.5",
    "x-ratelimit-reset-tokens-minute": "5.5",
}


def _chat_request() -> CreateChatCompletionRequest:
    return CreateChatCompletionRequest(
        model="gpt-oss-120b",
        messages=(UserMessage(role="user", content="hi"),),
        reasoning_format=ReasoningFormat.PARSED,
    )


def _completion_json() -> dict[str, object]:
    return {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello there",
                    "reasoning": "thinking",
                },
                "finish_reason": "stop",
            }
        ],
        "model": "gpt-oss-120b",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    return Client(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_empty_api_key_is_rejected():
    with pytest.raises(ValueError, match="api_key"):
        Client(api_key="", http_client=httpx.Client())


def test_chat_posts_exact_body_and_returns_complete_metadata():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json=_completion_json(),
            headers={
                "x-request-id": "req-1",
                "retry-after": "2.5",
                **RATE_LIMIT_HEADERS,
            },
            request=request,
        )

    result = _client(handler).create_chat_completion(_chat_request())

    assert captured == {
        "method": "POST",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "authorization": "Bearer test-key",
        "content_type": "application/json",
        "body": {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-oss-120b",
            "reasoning_format": "parsed",
        },
    }
    assert isinstance(result, APIResponse)
    assert isinstance(result.data, ChatCompletion)
    assert result.data.id == "chatcmpl-123"
    assert result.metadata.status_code == 201
    assert result.metadata.request_id == "req-1"
    assert result.metadata.retry_after == 2.5
    assert result.metadata.rate_limit.limit_requests_day == 100
    assert result.metadata.rate_limit.limit_tokens_minute == 1000
    assert result.metadata.rate_limit.remaining_requests_day == 99
    assert result.metadata.rate_limit.remaining_tokens_minute == 900
    assert result.metadata.rate_limit.reset_requests_day == 30.5
    assert result.metadata.rate_limit.reset_tokens_minute == 5.5


def test_missing_optional_metadata_is_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_json(), request=request)

    metadata = _client(handler).create_chat_completion(_chat_request()).metadata

    assert metadata.request_id is None
    assert metadata.retry_after is None
    assert metadata.rate_limit.limit_requests_day is None
    assert metadata.rate_limit.reset_tokens_minute is None


def test_zero_retry_after_is_preserved():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            text="slow",
            headers={"retry-after": "0"},
            request=request,
        )

    with pytest.raises(RateLimitError) as exc_info:
        _client(handler).create_chat_completion(_chat_request())

    assert exc_info.value.metadata.retry_after == 0.0


@pytest.mark.parametrize("status_code", [302, 400, 403, 500])
def test_other_non_success_statuses_map_to_exact_generic_api_error(
    status_code: int,
):
    body = "failure"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            text=body,
            headers={"x-request-id": f"req-{status_code}"},
            request=request,
        )

    with pytest.raises(APIError) as exc_info:
        _client(handler).create_chat_completion(_chat_request())

    error = exc_info.value
    assert type(error) is APIError
    assert error.body == body
    assert error.metadata.status_code == status_code
    assert error.metadata.request_id == f"req-{status_code}"


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(401, AuthError), (429, RateLimitError)],
)
def test_specialized_http_errors_retain_complete_response_evidence(
    status_code: int,
    error_type: type[AuthError | RateLimitError],
):
    body = "provider error"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            text=body,
            headers={
                "x-request-id": f"req-{status_code}",
                "retry-after": "2.5",
                **RATE_LIMIT_HEADERS,
            },
            request=request,
        )

    with pytest.raises(error_type) as exc_info:
        _client(handler).create_chat_completion(_chat_request())

    error = exc_info.value
    assert type(error) is error_type
    assert error.body == body
    assert error.metadata.status_code == status_code
    assert error.metadata.request_id == f"req-{status_code}"
    assert error.metadata.retry_after == 2.5
    assert error.metadata.rate_limit.limit_requests_day == 100
    assert error.metadata.rate_limit.remaining_tokens_minute == 900


def test_invalid_json_retains_decode_evidence():
    body = "not json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=body,
            headers={"x-request-id": "req-json"},
            request=request,
        )

    with pytest.raises(ResponseError) as exc_info:
        _client(handler).create_chat_completion(_chat_request())

    error = exc_info.value
    assert error.body == body
    assert error.metadata.request_id == "req-json"
    assert isinstance(error.__cause__, json.JSONDecodeError)
    assert error.__cause__.doc == body


def test_invalid_json_encoding_retains_decode_evidence():
    body = b"\xff"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, request=request)

    with pytest.raises(ResponseError) as exc_info:
        _client(handler).create_chat_completion(_chat_request())

    error = exc_info.value
    assert error.body == "\ufffd"
    assert isinstance(error.__cause__, UnicodeDecodeError)
    assert error.__cause__.object == body


def test_schema_invalid_success_retains_response_evidence():
    body = "[]"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"x-request-id": "req-schema"},
            request=request,
        )

    with pytest.raises(ResponseError) as exc_info:
        _client(handler).create_chat_completion(_chat_request())

    error = exc_info.value
    assert error.body == body
    assert error.metadata.status_code == 200
    assert error.metadata.request_id == "req-schema"
    assert isinstance(error.__cause__, ValidationError)


def test_empty_success_body_is_a_response_error_with_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            204,
            headers={"x-request-id": "req-empty"},
            request=request,
        )

    with pytest.raises(ResponseError) as exc_info:
        _client(handler).create_chat_completion(_chat_request())

    assert exc_info.value.metadata.status_code == 204
    assert exc_info.value.metadata.request_id == "req-empty"
    assert exc_info.value.body == ""


def test_transport_failure_retains_exact_cause_and_no_metadata():
    message = "connection refused"
    transport_error = httpx.ConnectError(message)

    def handler(request: httpx.Request) -> httpx.Response:
        transport_error.request = request
        raise transport_error

    with pytest.raises(TransportError) as exc_info:
        _client(handler).create_chat_completion(_chat_request())

    assert exc_info.value.__cause__ is transport_error
    assert exc_info.value.metadata is None


def test_timeout_failure_retains_exact_cause_and_no_metadata():
    message = "timed out"
    timeout = httpx.ReadTimeout(message)

    def handler(request: httpx.Request) -> httpx.Response:
        timeout.request = request
        raise timeout

    with pytest.raises(TransportError) as exc_info:
        _client(handler).create_chat_completion(_chat_request())

    assert exc_info.value.__cause__ is timeout
    assert exc_info.value.metadata is None


@pytest.mark.parametrize(
    ("case", "error_type"),
    [
        ("success", None),
        ("auth", AuthError),
        ("rate_limit", RateLimitError),
        ("api", APIError),
        ("invalid_json", ResponseError),
        ("invalid_schema", ResponseError),
        ("transport", TransportError),
    ],
)
def test_client_remains_open_and_never_retries(
    case: str,
    error_type: type[Exception] | None,
):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if case == "success":
            return httpx.Response(200, json=_completion_json(), request=request)
        if case == "auth":
            return httpx.Response(401, text="auth", request=request)
        if case == "rate_limit":
            return httpx.Response(429, text="slow", request=request)
        if case == "api":
            return httpx.Response(500, text="server", request=request)
        if case == "invalid_json":
            return httpx.Response(200, text="not json", request=request)
        if case == "invalid_schema":
            return httpx.Response(200, json=[], request=request)
        message = "connection refused"
        raise httpx.ConnectError(message, request=request)

    injected = httpx.Client(transport=httpx.MockTransport(handler))
    client = Client(api_key="test-key", http_client=injected)

    if error_type is None:
        client.create_chat_completion(_chat_request())
    else:
        with pytest.raises(error_type):
            client.create_chat_completion(_chat_request())

    assert attempts == 1
    assert injected.is_closed is False
