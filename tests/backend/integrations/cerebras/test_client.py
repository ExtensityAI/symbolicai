import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.integrations.cerebras.chat import (
    ChatRequest,
    ChatResponse,
    ReasoningFormat,
    UserMessage,
)
from symai.backend.integrations.cerebras.client import Client
from symai.backend.integrations.cerebras.errors import (
    APIError,
    AuthError,
    RateLimitError,
    ResponseError,
    TransportError,
)
from symai.backend.integrations.cerebras.response import Response

RATE_LIMIT_HEADERS = {
    "x-ratelimit-limit-requests-day": "100",
    "x-ratelimit-limit-tokens-minute": "1000",
    "x-ratelimit-remaining-requests-day": "99",
    "x-ratelimit-remaining-tokens-minute": "900",
    "x-ratelimit-reset-requests-day": "30.5",
    "x-ratelimit-reset-tokens-minute": "5.5",
}


def _chat_request() -> ChatRequest:
    return ChatRequest(
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


def test_nonempty_api_key_is_preserved_exactly():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        return httpx.Response(200, json=_completion_json(), request=request)

    injected = httpx.Client(transport=httpx.MockTransport(handler))
    Client(api_key=" test-key ", http_client=injected).chat(_chat_request())

    assert captured["authorization"] == "Bearer  test-key "


def test_chat_posts_exact_body_and_returns_complete_metadata():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
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

    result = _client(handler).chat(_chat_request())

    assert captured == {
        "method": "POST",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "authorization": "Bearer test-key",
        "body": {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-oss-120b",
            "reasoning_format": "parsed",
        },
    }
    assert isinstance(result, Response)
    assert isinstance(result.data, ChatResponse)
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

    metadata = _client(handler).chat(_chat_request()).metadata

    assert metadata.request_id is None
    assert metadata.retry_after is None
    assert metadata.rate_limit.limit_requests_day is None
    assert metadata.rate_limit.reset_tokens_minute is None


@pytest.mark.parametrize(
    "value",
    ["", "invalid", "-1", "nan", "inf", "-inf", "1_0"],
)
def test_invalid_numeric_metadata_becomes_none(value: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion_json(),
            headers={
                "retry-after": value,
                "x-ratelimit-limit-requests-day": value,
                "x-ratelimit-reset-tokens-minute": value,
            },
            request=request,
        )

    metadata = _client(handler).chat(_chat_request()).metadata

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
        _client(handler).chat(_chat_request())

    assert exc_info.value.retry_after == 0.0


def test_retry_after_http_date_is_not_guessed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            text="slow",
            headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
            request=request,
        )

    with pytest.raises(RateLimitError) as exc_info:
        _client(handler).chat(_chat_request())

    assert exc_info.value.retry_after is None


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(401, AuthError), (429, RateLimitError), (500, APIError)],
)
def test_http_statuses_map_to_typed_errors(
    status_code: int,
    error_type: type[APIError],
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="failure", request=request)

    with pytest.raises(error_type):
        _client(handler).chat(_chat_request())


def test_http_error_retains_complete_metadata_and_exact_body():
    body = "server error"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            text=body,
            headers={"x-request-id": "req-500", **RATE_LIMIT_HEADERS},
            request=request,
        )

    with pytest.raises(APIError) as exc_info:
        _client(handler).chat(_chat_request())

    error = exc_info.value
    assert error.body == body
    assert error.status_code == 500
    assert error.request_id == "req-500"
    assert error.metadata.rate_limit.remaining_tokens_minute == 900


def test_invalid_json_retains_exact_decode_cause(
    monkeypatch: pytest.MonkeyPatch,
):
    body = "not json"
    decode_error = json.JSONDecodeError("invalid", body, 0)

    def fail_json(**_kwargs: object) -> object:
        raise decode_error

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            text=body,
            headers={"x-request-id": "req-json"},
            request=request,
        )
        monkeypatch.setattr(response, "json", fail_json)
        return response

    with pytest.raises(ResponseError) as exc_info:
        _client(handler).chat(_chat_request())

    error = exc_info.value
    assert error.body == body
    assert error.metadata.request_id == "req-json"
    assert error.__cause__ is decode_error


def test_invalid_json_encoding_retains_exact_decode_cause(
    monkeypatch: pytest.MonkeyPatch,
):
    body = "invalid encoding"
    decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, body)

    def fail_json(**_kwargs: object) -> object:
        raise decode_error

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, text=body, request=request)
        monkeypatch.setattr(response, "json", fail_json)
        return response

    with pytest.raises(ResponseError) as exc_info:
        _client(handler).chat(_chat_request())

    assert exc_info.value.__cause__ is decode_error


def test_schema_invalid_success_retains_exact_validation_cause(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(ValidationError) as source:
        ChatResponse.model_validate([])
    validation_error = source.value

    def fail_validation(_payload: object) -> ChatResponse:
        raise validation_error

    monkeypatch.setattr(ChatResponse, "model_validate", fail_validation)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={},
            headers={"x-request-id": "req-schema"},
            request=request,
        )

    with pytest.raises(ResponseError) as exc_info:
        _client(handler).chat(_chat_request())

    assert exc_info.value.metadata.request_id == "req-schema"
    assert exc_info.value.__cause__ is validation_error


def test_empty_success_body_is_a_response_error_with_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            204,
            headers={"x-request-id": "req-empty"},
            request=request,
        )

    with pytest.raises(ResponseError) as exc_info:
        _client(handler).chat(_chat_request())

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
        _client(handler).chat(_chat_request())

    assert exc_info.value.__cause__ is transport_error
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
        client.chat(_chat_request())
    else:
        with pytest.raises(error_type):
            client.chat(_chat_request())

    assert attempts == 1
    assert injected.is_closed is False
