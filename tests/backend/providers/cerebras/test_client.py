import json

import httpx
import pytest

from symai.backend.providers.cerebras.client import Client
from symai.backend.providers.cerebras.errors import (
    APIError,
    AuthError,
    RateLimitError,
    ResponseError,
    TransportError,
)
from symai.backend.providers.cerebras.request import (
    ChatRequest,
    JsonSchemaSpec,
    Message,
    ResponseFormat,
    Role,
)
from symai.backend.providers.cerebras.response import ChatResponse
from symai.backend.providers.cerebras.spec import Model


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


def _client_with_response(status_code: int, **response_kwargs) -> Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request, **response_kwargs)

    return Client(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


# --- create() happy path -------------------------------------------------------


def test_create_success_posts_expected_request_and_parses_response():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_completion_json(), request=request)

    client = Client(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    response = client.create(_chat_request())

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/chat/completions")
    assert captured["authorization"] == "Bearer test-key"
    assert "model" in captured["body"]
    assert captured["body"]["messages"]
    assert captured["body"]["response_format"]["json_schema"]["schema"] == {"type": "object"}

    assert isinstance(response, ChatResponse)
    assert response.choices[0].message.content == "hello there"
    assert response.usage.total_tokens == 15


# --- status/body -> typed error mapping -----------------------------------------


def test_error_401_raises_auth_error():
    client = _client_with_response(401, text="invalid api key")

    with pytest.raises(AuthError):
        client.create(_chat_request())


def test_error_429_raises_rate_limit_error():
    client = _client_with_response(429, text="rate limited")

    with pytest.raises(RateLimitError):
        client.create(_chat_request())


def test_error_500_raises_api_error_with_status_and_body():
    body_text = "internal server error"
    client = _client_with_response(500, text=body_text)

    with pytest.raises(APIError) as exc_info:
        client.create(_chat_request())

    assert exc_info.value.status_code == 500
    assert exc_info.value.body == body_text


def test_error_malformed_response_raises_response_error():
    payload = _completion_json()
    del payload["usage"]
    body_text = json.dumps(payload)
    client = _client_with_response(200, content=body_text)

    with pytest.raises(ResponseError) as exc_info:
        client.create(_chat_request())

    assert exc_info.value.body == body_text


def test_error_invalid_json_response_raises_response_error():
    body_text = "not json at all"
    client = _client_with_response(200, content=body_text)

    with pytest.raises(ResponseError) as exc_info:
        client.create(_chat_request())

    assert exc_info.value.body == body_text


def test_connection_failure_raises_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = Client(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(TransportError):
        client.create(_chat_request())


# --- client lifecycle -----------------------------------------------------------


def test_owned_http_client_is_closed_on_context_exit():
    with Client(api_key="test-key") as client:
        assert client._http_client.is_closed is False

    assert client._http_client.is_closed is True


def test_injected_http_client_is_not_closed_by_close():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_json(), request=request)

    injected = httpx.Client(transport=httpx.MockTransport(handler))

    with Client(api_key="test-key", http_client=injected) as client:
        assert client._http_client.is_closed is False

    assert injected.is_closed is False
