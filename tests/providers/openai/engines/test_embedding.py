import json
from collections.abc import Callable
from typing import cast

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from symai.providers.openai import EmbeddingEngine
from symai.providers.openai.client import Client
from symai.providers.openai.client import errors as openai_errors
from symai.providers.openai.client.embeddings import CreateEmbeddingRequest
from symai.providers.openai.client.transport import ResponseMetadata as OpenAIResponseMetadata
from symai.runtime.errors import (
    AuthenticationError,
    ExecutionError,
    InvalidResponseError,
    RateLimitError,
    TransportError,
    UnsupportedFeatureError,
    UnsupportedModelError,
)
from symai.runtime.models import EmbeddingRequest


def _embedding_json(
    *,
    data: list[dict[str, object]] | None = None,
    prompt_tokens: int = 2,
    total_tokens: int = 2,
    model: str = "text-embedding-3-small",
) -> dict[str, object]:
    return {
        "object": "list",
        "data": data
        if data is not None
        else [
            {"object": "embedding", "embedding": [1.0, 0.0], "index": 0},
            {"object": "embedding", "embedding": [0.0, 1.0], "index": 1},
        ],
        "model": model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": total_tokens},
    }


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Client:
    return Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )


def test_execute_translates_request_and_sorts_provider_vectors() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.read()))
        return httpx.Response(
            200,
            headers={"x-request-id": "request-id", "retry-after": "1.25"},
            json=_embedding_json(
                data=[
                    {"object": "embedding", "embedding": [0.0, 1.0], "index": 1},
                    {"object": "embedding", "embedding": [1.0, 0.0], "index": 0},
                ]
            ),
        )

    client = _client(handler)
    engine = EmbeddingEngine(client=client, model="text-embedding-3-small")
    try:
        response = engine.execute(
            EmbeddingRequest(inputs=("one", "two"), dimensions=2, user="customer-42")
        )
    finally:
        engine.close()

    assert captured_body == {
        "input": ["one", "two"],
        "model": "text-embedding-3-small",
        "dimensions": 2,
        "encoding_format": "float",
        "user": "customer-42",
    }
    assert tuple(vector.index for vector in response.vectors) == (0, 1)
    assert tuple(vector.values for vector in response.vectors) == ((1.0, 0.0), (0.0, 1.0))
    assert response.metadata.provider == "openai"
    assert response.metadata.requested_model == "text-embedding-3-small"
    assert response.metadata.response_model == "text-embedding-3-small"
    assert response.metadata.status_code == 200
    assert response.metadata.request_id == "request-id"
    assert response.metadata.retry_after == 1.25
    assert response.metadata.usage is not None
    assert response.metadata.usage.prompt_tokens == 2
    assert response.metadata.usage.total_tokens == 2
    assert response.metadata.usage.completion_tokens == 0


def test_response_model_identity_is_preserved_without_prefix_matching() -> None:
    returned_model = "provider-resolved-embedding-model-2026-07-15"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_embedding_json(model=returned_model))

    client = _client(handler)
    engine = EmbeddingEngine(client=client, model="text-embedding-3-small")
    try:
        response = engine.execute(EmbeddingRequest(inputs=("one", "two"), dimensions=2))
    finally:
        engine.close()

    assert response.metadata.requested_model == "text-embedding-3-small"
    assert response.metadata.response_model == returned_model


def test_default_dimensions_are_omitted_from_provider_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.read()) == {
            "input": ["one", "two"],
            "model": "text-embedding-3-large",
            "encoding_format": "float",
        }
        return httpx.Response(
            200,
            json={
                **_embedding_json(),
                "model": "text-embedding-3-large",
                "data": [
                    {"object": "embedding", "embedding": [1.0] * 3072, "index": 0},
                    {"object": "embedding", "embedding": [0.0] * 3072, "index": 1},
                ],
            },
        )

    client = _client(handler)
    engine = EmbeddingEngine(client=client, model="text-embedding-3-large")
    try:
        response = engine.execute(EmbeddingRequest(inputs=("one", "two")))
    finally:
        engine.close()

    assert len(response.vectors[0].values) == 3072


@pytest.mark.parametrize(
    ("model", "dimensions"),
    [
        ("text-embedding-ada-002", 128),
        ("text-embedding-3-small", 1537),
        ("text-embedding-3-large", 3073),
    ],
)
def test_invalid_model_dimension_combination_fails_before_client_invocation(
    model: str,
    dimensions: int,
) -> None:
    calls = 0

    class RecordingClient:
        def create_embeddings(self, _request: CreateEmbeddingRequest):
            nonlocal calls
            calls += 1
            msg = "client must not be called"
            raise AssertionError(msg)

    engine = EmbeddingEngine(client=cast("Client", RecordingClient()), model=model)

    with pytest.raises(UnsupportedFeatureError, match="dimensions"):
        engine.execute(EmbeddingRequest(inputs=("hello",), dimensions=dimensions))

    assert calls == 0


def test_unknown_model_is_rejected_at_construction() -> None:
    with pytest.raises(UnsupportedModelError, match="future-model"):
        EmbeddingEngine(
            client=cast("Client", object()),
            model="future-model",
        )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            [
                {"object": "embedding", "embedding": [1.0, 0.0], "index": 0},
                {"object": "embedding", "embedding": [0.0, 1.0], "index": 0},
            ],
            "duplicate",
        ),
        (
            [{"object": "embedding", "embedding": [1.0, 0.0], "index": 1}],
            "indices",
        ),
        (
            [
                {"object": "embedding", "embedding": [1.0, 0.0, 2.0], "index": 0},
                {"object": "embedding", "embedding": [0.0, 1.0], "index": 1},
            ],
            "dimensions",
        ),
        (
            [
                {"object": "embedding", "embedding": "base64", "index": 0},
                {"object": "embedding", "embedding": [0.0, 1.0], "index": 1},
            ],
            "float",
        ),
        (
            [
                {"object": "embedding", "embedding": [float("nan"), 0.0], "index": 0},
                {"object": "embedding", "embedding": [0.0, 1.0], "index": 1},
            ],
            "normalized embedding response",
        ),
    ],
)
def test_malformed_embedding_data_is_rejected(
    data: list[dict[str, object]],
    message: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(_embedding_json(data=data)).encode())

    client = _client(handler)
    engine = EmbeddingEngine(client=client, model="text-embedding-3-small")
    try:
        with pytest.raises(InvalidResponseError, match=message):
            engine.execute(EmbeddingRequest(inputs=("one", "two"), dimensions=2))
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("prompt_tokens", "total_tokens"),
    [(2, 3), (-1, -1)],
)
def test_inconsistent_usage_is_omitted(prompt_tokens: int, total_tokens: int) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_embedding_json(
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            ),
        )

    client = _client(handler)
    engine = EmbeddingEngine(client=client, model="text-embedding-3-small")
    try:
        response = engine.execute(EmbeddingRequest(inputs=("one", "two"), dimensions=2))
    finally:
        engine.close()

    assert tuple(vector.index for vector in response.vectors) == (0, 1)
    assert response.metadata.usage is None


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
        (openai_errors.TransportError("network failed"), TransportError),
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
        def create_embeddings(self, _request: CreateEmbeddingRequest):
            raise provider_error

    engine = EmbeddingEngine(
        client=cast("Client", FailingClient()),
        model="text-embedding-3-small",
    )

    with pytest.raises(runtime_error) as caught:
        engine.execute(EmbeddingRequest(inputs=("hello",), dimensions=2))

    assert caught.value.__cause__ is provider_error
    assert "secret body" not in str(caught.value)
    if isinstance(provider_error, (openai_errors.APIError, openai_errors.ResponseError)):
        assert caught.value.metadata is not None
        assert caught.value.metadata.request_id == provider_error.metadata.request_id
        assert caught.value.metadata.retry_after == provider_error.metadata.retry_after


def test_invalid_provider_metadata_is_normalized() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"retry-after": "inf"},
            json=_embedding_json(),
        )

    client = _client(handler)
    engine = EmbeddingEngine(client=client, model="text-embedding-3-small")
    try:
        response = engine.execute(EmbeddingRequest(inputs=("one", "two"), dimensions=2))
    finally:
        engine.close()

    assert response.metadata.retry_after is None


def test_normalized_validation_failure_is_chained() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(
                _embedding_json(
                    data=[{"object": "embedding", "embedding": [float("inf")], "index": 0}]
                )
            ).encode(),
        )

    client = _client(handler)
    engine = EmbeddingEngine(client=client, model="text-embedding-3-small")
    try:
        with pytest.raises(InvalidResponseError) as caught:
            engine.execute(EmbeddingRequest(inputs=("one",), dimensions=1))
    finally:
        engine.close()

    assert isinstance(caught.value.__cause__, ValidationError)
