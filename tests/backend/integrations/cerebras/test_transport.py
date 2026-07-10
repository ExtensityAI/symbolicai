import json

import httpx
import pytest

from symai.backend.integrations.base import StrictModel, TolerantModel
from symai.backend.integrations.cerebras.client.errors import (
    APIError,
    AuthError,
    RateLimitError,
    ResponseError,
    TransportError,
)
from symai.backend.integrations.cerebras.client.transport import Transport

# The transport knows nothing of any endpoint, so these stand in for one.


class _Request(StrictModel):
    question: str


class _Reply(TolerantModel):
    answer: str


def _reply_json() -> dict:
    return {"answer": "42"}


def _transport(handler) -> Transport:
    return Transport(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def _transport_with_response(status_code: int, **response_kwargs) -> Transport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request, **response_kwargs)

    return _transport(handler)


def _post(transport: Transport) -> _Reply:
    return transport.post("/ask", _Request(question="what"), _Reply)


# --- construction ---------------------------------------------------------------


def test_empty_api_key_is_rejected():
    with pytest.raises(ValueError, match="api_key"):
        Transport(api_key="", http_client=httpx.Client())


# --- post() happy path ----------------------------------------------------------


def test_post_sends_credentials_to_the_base_url_and_validates_the_reply():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_reply_json(), request=request)

    reply = _post(_transport(handler))

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.cerebras.ai/v1/ask"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"] == {"question": "what"}

    assert isinstance(reply, _Reply)
    assert reply.answer == "42"


# --- status/body -> typed error mapping -----------------------------------------


def test_error_401_raises_auth_error():
    with pytest.raises(AuthError):
        _post(_transport_with_response(401, text="invalid api key"))


def test_error_429_raises_rate_limit_error():
    with pytest.raises(RateLimitError):
        _post(_transport_with_response(429, text="rate limited"))


def test_error_500_raises_api_error_with_status_and_body():
    body_text = "internal server error"

    with pytest.raises(APIError) as exc_info:
        _post(_transport_with_response(500, text=body_text))

    assert exc_info.value.status_code == 500
    assert exc_info.value.body == body_text


def test_error_malformed_response_raises_response_error():
    body_text = json.dumps({"unexpected": "shape"})

    with pytest.raises(ResponseError) as exc_info:
        _post(_transport_with_response(200, content=body_text))

    assert exc_info.value.body == body_text


def test_error_invalid_json_response_raises_response_error():
    body_text = "not json at all"

    with pytest.raises(ResponseError) as exc_info:
        _post(_transport_with_response(200, content=body_text))

    assert exc_info.value.body == body_text


def test_connection_failure_raises_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(TransportError):
        _post(_transport(handler))


# --- API-instruction headers ------------------------------------------------------


def test_error_surfaces_request_id_header():
    transport = _transport_with_response(500, text="boom", headers={"x-request-id": "req-abc"})

    with pytest.raises(APIError) as exc_info:
        _post(transport)

    assert exc_info.value.request_id == "req-abc"


def test_rate_limit_surfaces_retry_after_seconds():
    transport = _transport_with_response(
        429, text="slow", headers={"retry-after": "3", "x-request-id": "req-xyz"}
    )

    with pytest.raises(RateLimitError) as exc_info:
        _post(transport)

    assert exc_info.value.retry_after == 3.0
    assert exc_info.value.request_id == "req-xyz"


def test_rate_limit_without_retry_after_header_is_none():
    with pytest.raises(RateLimitError) as exc_info:
        _post(_transport_with_response(429, text="slow"))

    assert exc_info.value.retry_after is None


def test_retry_after_http_date_yields_none_rather_than_a_guess():
    transport = _transport_with_response(
        429, text="slow", headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    )

    with pytest.raises(RateLimitError) as exc_info:
        _post(transport)

    assert exc_info.value.retry_after is None


# --- transport ownership ----------------------------------------------------------


def test_transport_never_closes_the_httpx_client_it_does_not_own():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reply_json(), request=request)

    injected = httpx.Client(transport=httpx.MockTransport(handler))

    Transport(api_key="test-key", http_client=injected).post(
        "/ask", _Request(question="what"), _Reply
    )

    assert injected.is_closed is False
