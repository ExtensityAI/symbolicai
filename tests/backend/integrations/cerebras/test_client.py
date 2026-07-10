import json

import httpx
import pytest

from symai.backend.integrations.cerebras.chat import (
    ChatRequest,
    ChatResponse,
    JsonSchemaSpec,
    Message,
    ResponseFormat,
    Role,
)
from symai.backend.integrations.cerebras.client import Client
from symai.backend.integrations.cerebras.errors import (
    APIError,
    AuthError,
    RateLimitError,
    ResponseError,
    TransportError,
)
from symai.backend.integrations.cerebras.models import Model


def _chat_request() -> ChatRequest:
    schema_spec = JsonSchemaSpec(name="Answer", json_schema_body={"type": "object"})
    response_format = ResponseFormat(type="json_schema", json_schema=schema_spec)
    return ChatRequest(
        messages=(Message(role=Role.USER, content="hi"),),
        model=Model.GPT_OSS_120B,
        response_format=response_format,
    )


def _completion_json() -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _client(handler) -> Client:
    return Client(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def _client_with_response(status_code: int, **response_kwargs) -> Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request, **response_kwargs)

    return _client(handler)


# --- construction ---------------------------------------------------------------


def test_empty_api_key_is_rejected():
    with pytest.raises(ValueError, match="api_key"):
        Client(api_key="", http_client=httpx.Client())


# --- chat() happy path ----------------------------------------------------------


def test_chat_posts_to_the_completions_path_and_parses_the_response():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_completion_json(), request=request)

    response = _client(handler).chat(_chat_request())

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.cerebras.ai/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == Model.GPT_OSS_120B
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["body"]["response_format"]["json_schema"]["schema"] == {"type": "object"}

    assert isinstance(response, ChatResponse)
    assert response.choices[0].message.content == "hello there"
    assert response.usage.total_tokens == 15


# --- status/body -> typed error mapping -----------------------------------------


def test_error_401_raises_auth_error():
    with pytest.raises(AuthError):
        _client_with_response(401, text="invalid api key").chat(_chat_request())


def test_error_429_raises_rate_limit_error():
    with pytest.raises(RateLimitError):
        _client_with_response(429, text="rate limited").chat(_chat_request())


def test_error_500_raises_api_error_with_status_and_body():
    body_text = "internal server error"

    with pytest.raises(APIError) as exc_info:
        _client_with_response(500, text=body_text).chat(_chat_request())

    assert exc_info.value.status_code == 500
    assert exc_info.value.body == body_text


def test_error_malformed_response_raises_response_error():
    payload = _completion_json()
    del payload["usage"]
    body_text = json.dumps(payload)

    with pytest.raises(ResponseError) as exc_info:
        _client_with_response(200, content=body_text).chat(_chat_request())

    assert exc_info.value.body == body_text


def test_error_invalid_json_response_raises_response_error():
    body_text = "not json at all"

    with pytest.raises(ResponseError) as exc_info:
        _client_with_response(200, content=body_text).chat(_chat_request())

    assert exc_info.value.body == body_text


def test_connection_failure_raises_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(TransportError):
        _client(handler).chat(_chat_request())


# --- API-instruction headers ------------------------------------------------------


def test_error_surfaces_request_id_header():
    client = _client_with_response(500, text="boom", headers={"x-request-id": "req-abc"})

    with pytest.raises(APIError) as exc_info:
        client.chat(_chat_request())

    assert exc_info.value.request_id == "req-abc"


def test_rate_limit_surfaces_retry_after_seconds():
    client = _client_with_response(
        429, text="slow", headers={"retry-after": "3", "x-request-id": "req-xyz"}
    )

    with pytest.raises(RateLimitError) as exc_info:
        client.chat(_chat_request())

    assert exc_info.value.retry_after == 3.0
    assert exc_info.value.request_id == "req-xyz"


def test_rate_limit_without_retry_after_header_is_none():
    with pytest.raises(RateLimitError) as exc_info:
        _client_with_response(429, text="slow").chat(_chat_request())

    assert exc_info.value.retry_after is None


def test_retry_after_http_date_yields_none_rather_than_a_guess():
    client = _client_with_response(
        429, text="slow", headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    )

    with pytest.raises(RateLimitError) as exc_info:
        client.chat(_chat_request())

    assert exc_info.value.retry_after is None


# --- transport ownership ----------------------------------------------------------


def test_client_never_closes_the_httpx_client_it_does_not_own():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_json(), request=request)

    injected = httpx.Client(transport=httpx.MockTransport(handler))

    Client(api_key="test-key", http_client=injected).chat(_chat_request())

    assert injected.is_closed is False
