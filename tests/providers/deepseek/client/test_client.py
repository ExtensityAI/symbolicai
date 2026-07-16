import json

import httpx
import pytest
from pydantic import BaseModel, Field, SecretStr

from symai.providers._http.headers import _UNSAFE_API_KEY_MESSAGE
from symai.providers.deepseek.client.chat import CreateChatCompletionRequest, UserMessage
from symai.providers.deepseek.client.client import Client
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

    client = Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )
    try:
        response = client.create_chat_completion(_request())
    finally:
        client.close()

    assert response.data.choices[0].message.content == "hi"
    assert response.metadata.status_code == 200
    assert response.metadata.request_id == "request-id"
    assert isinstance(response, BaseModel)
    assert isinstance(response.metadata, BaseModel)


def test_chat_serializes_request_fields_by_alias() -> None:
    class RequestWithAlias(CreateChatCompletionRequest):
        future_option: str = Field(serialization_alias="futureOption")

    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(500, text="stop after serialization")

    client = Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )
    request = RequestWithAlias(
        messages=(UserMessage(role="user", content="hello"),),
        model="deepseek-v4-flash",
        future_option="enabled",
    )
    try:
        with pytest.raises(APIError):
            client.create_chat_completion(request)
    finally:
        client.close()

    assert captured_body["futureOption"] == "enabled"
    assert "future_option" not in captured_body


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

    client = Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(error_type) as raised:
            client.create_chat_completion(_request())
    finally:
        client.close()

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

    client = Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ResponseError) as raised:
            client.create_chat_completion(_request())
    finally:
        client.close()

    assert raised.value.metadata.status_code == 200
    assert raised.value.body == body


def test_chat_maps_transport_failures():
    def handler(request: httpx.Request):
        message = "offline"
        raise httpx.ConnectError(message, request=request)

    client = Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(TransportError) as raised:
            client.create_chat_completion(_request())
    finally:
        client.close()

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

    with pytest.raises(ValueError) as exc_info:
        Client(
            api_key=SecretStr(api_key),
            transport=httpx.MockTransport(handler),
        )

    assert attempts == 0
    # A constant message cannot be derived from the credential, and every invalid
    # credential must fail identically: which rule a key trips is a property of the secret.
    assert exc_info.value.args == (_UNSAFE_API_KEY_MESSAGE,)


def test_client_rejects_plaintext_api_key():
    with pytest.raises(TypeError) as exc_info:
        Client(
            api_key="test-key",  # pyright: ignore[reportArgumentType]
            transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
        )

    assert exc_info.value.args == ("api_key must be a SecretStr",)
