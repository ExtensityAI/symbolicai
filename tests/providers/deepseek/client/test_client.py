import httpx
import pytest
from pydantic import BaseModel, SecretStr

from symai.providers.deepseek.client.chat import CreateChatCompletionRequest, UserMessage
from symai.providers.deepseek.client import Client
from symai.providers.deepseek.client.errors import (
    APIError,
    AuthError,
    RateLimitError,
    ResponseError,
    TransportError,
)


def _request():
    return CreateChatCompletionRequest(
        messages=(UserMessage(role="user", content="hello"),),
        model="deepseek-v4-flash",
    )


def test_chat_posts_authenticated_request_and_returns_metadata():
    def handler(request: httpx.Request):
        assert request.method == "POST"
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["content-type"] == "application/json"
        assert request.read() == (
            b'{"messages":[{"role":"user","content":"hello"}],"model":"deepseek-v4-flash"}'
        )
        return httpx.Response(
            200,
            headers={"x-request-id": "request-id"},
            json={
                "id": "response-id",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "hi", "role": "assistant"},
                    }
                ],
                "created": 1,
                "model": "deepseek-v4-flash",
                "object": "chat.completion",
                "usage": {
                    "completion_tokens": 1,
                    "prompt_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        response = Client(
            api_key=SecretStr("test-key"), http_client=http_client
        ).create_chat_completion(_request())

    assert response.data.choices[0].message.content == "hi"
    assert response.metadata.status_code == 200
    assert response.metadata.request_id == "request-id"
    assert isinstance(response, BaseModel)
    assert isinstance(response.metadata, BaseModel)


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, AuthError), (429, RateLimitError), (500, APIError)],
)
def test_chat_maps_http_failures(status, error_type):
    def handler(_request: httpx.Request):
        return httpx.Response(
            status,
            headers={"x-request-id": "request-id", "retry-after": "2.5"},
            text="failure body",
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as http_client,
        pytest.raises(error_type) as raised,
    ):
        Client(api_key=SecretStr("test-key"), http_client=http_client).create_chat_completion(
            _request()
        )

    assert raised.value.metadata.status_code == status
    assert raised.value.metadata.request_id == "request-id"
    assert raised.value.body == "failure body"
    if status == 429:
        assert raised.value.metadata.retry_after == 2.5


@pytest.mark.parametrize(
    ("body", "content_type"),
    [("not json", "text/plain"), ('{"choices": "wrong"}', "application/json")],
)
def test_chat_maps_invalid_success_responses(body, content_type):
    def handler(_request: httpx.Request):
        return httpx.Response(200, text=body, headers={"content-type": content_type})

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as http_client,
        pytest.raises(ResponseError) as raised,
    ):
        Client(api_key=SecretStr("test-key"), http_client=http_client).create_chat_completion(
            _request()
        )

    assert raised.value.metadata.status_code == 200
    assert raised.value.body == body


def test_chat_maps_transport_failures():
    def handler(request: httpx.Request):
        message = "offline"
        raise httpx.ConnectError(message, request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as http_client,
        pytest.raises(TransportError) as raised,
    ):
        Client(api_key=SecretStr("test-key"), http_client=http_client).create_chat_completion(
            _request()
        )

    assert raised.value.__cause__.__class__ is httpx.ConnectError
    assert raised.value.metadata is None


@pytest.mark.parametrize(
    "api_key",
    [
        "",
        "secret\rvalue",
        "secret\nvalue",
        "secret\x01value",
        "secret\x7fvalue",
        " secret",
        "secret ",
        "api_key ",
        " ",
        "ValueError\n",
        "TypeError\n",
    ],
    ids=[
        "empty",
        "cr",
        "lf",
        "c0",
        "del",
        "leading-space",
        "trailing-space",
        "message-collision",
        "whitespace-only",
        "value-error-traceback-text",
        "type-error-traceback-text",
    ],
)
def test_client_rejects_unsafe_api_key_before_request_without_disclosure(
    api_key: str,
):
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as http_client,
        pytest.raises(ValueError) as exc_info,
    ):
        Client(api_key=SecretStr(api_key), http_client=http_client)

    assert attempts == 0
    assert exc_info.value.args == ()
    assert str(exc_info.value) == ""


def test_client_rejects_plaintext_api_key():
    with (
        httpx.Client() as http_client,
        pytest.raises(TypeError) as exc_info,
    ):
        Client(
            api_key="test-key",  # pyright: ignore[reportArgumentType]
            http_client=http_client,
        )

    assert exc_info.value.args == ()
    assert str(exc_info.value) == ""
