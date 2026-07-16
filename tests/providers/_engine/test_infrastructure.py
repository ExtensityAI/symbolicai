from math import nan

import pytest

from symai.providers._engine.base import ProviderEngine
from symai.providers._engine.mapping import ClientErrorMessages, raise_mapped_client_error
from symai.providers._http import errors as client_errors
from symai.providers._http.response import HttpMetadata
from symai.runtime.errors import (
    AuthenticationError,
    ExecutionError,
    InvalidResponseError,
    RateLimitError,
    TransportError,
    UnsupportedModelError,
)


class RecordingClient:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self.close_count = 0
        self.close_error = close_error

    def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


def test_base_engine_owns_model_spec_and_closes_client_once() -> None:
    client = RecordingClient()
    engine = ProviderEngine(
        client=client,
        model="known-model",
        model_specs={"known-model": 17},
        unsupported_model_message="Unsupported test model: {model}",
    )

    engine.close()
    engine.close()

    assert engine.model == "known-model"
    assert engine.model_spec == 17
    assert client.close_count == 1


def test_base_engine_construction_cleanup_preserves_primary_failure() -> None:
    cleanup_error = KeyboardInterrupt("cleanup failed")
    client = RecordingClient(cleanup_error)

    with pytest.raises(UnsupportedModelError, match="missing-model") as caught:
        ProviderEngine(
            client=client,
            model="missing-model",
            model_specs={"known-model": 17},
            unsupported_model_message="Unsupported test model: {model}",
        )

    assert client.close_count == 1
    assert caught.value.__notes__ == [
        "Engine construction cleanup failed: KeyboardInterrupt('cleanup failed')"
    ]


_ERROR_MESSAGES = ClientErrorMessages(
    authentication="Provider rejected authentication",
    rate_limit="Provider rate-limited the request",
    response="Provider returned an invalid response",
    transport="Provider transport failed",
    api="Provider API request failed with status {status_code}",
)


@pytest.mark.parametrize(
    ("client_error", "runtime_error", "message"),
    [
        (
            client_errors.AuthError(
                HttpMetadata(status_code=401, request_id="auth-id", retry_after=None),
                "secret",
            ),
            AuthenticationError,
            "Provider rejected authentication",
        ),
        (
            client_errors.RateLimitError(
                HttpMetadata(status_code=429, request_id="rate-id", retry_after=2.5),
                "secret",
            ),
            RateLimitError,
            "Provider rate-limited the request",
        ),
        (
            client_errors.ResponseError(
                "response",
                metadata=HttpMetadata(
                    status_code=200,
                    request_id="response-id",
                    retry_after=None,
                ),
                body="secret",
            ),
            InvalidResponseError,
            "Provider returned an invalid response",
        ),
        (
            client_errors.TransportError("network"),
            TransportError,
            "Provider transport failed",
        ),
        (
            client_errors.APIError(
                HttpMetadata(status_code=500, request_id="api-id", retry_after=None),
                "secret",
            ),
            ExecutionError,
            "Provider API request failed with status 500",
        ),
    ],
)
def test_client_error_mapper_preserves_order_metadata_and_cause(
    client_error: client_errors.ClientError,
    runtime_error: type[ExecutionError],
    message: str,
) -> None:
    with pytest.raises(runtime_error, match=message) as caught:
        raise_mapped_client_error(
            client_error,
            provider="test-provider",
            model="test-model",
            messages=_ERROR_MESSAGES,
        )

    assert caught.value.__cause__ is client_error
    assert caught.value.metadata is not None
    assert caught.value.metadata.provider == "test-provider"
    assert caught.value.metadata.model == "test-model"
    metadata = getattr(client_error, "metadata", None)
    assert caught.value.metadata.request_id == (
        metadata.request_id if metadata is not None else None
    )
    assert caught.value.metadata.retry_after == (
        metadata.retry_after if metadata is not None else None
    )


@pytest.mark.parametrize("retry_after", [-1.0, nan])
def test_client_error_mapper_drops_invalid_retry_after(retry_after: float) -> None:
    error = client_errors.APIError(
        HttpMetadata(status_code=503, request_id="request-id", retry_after=retry_after),
        "secret",
    )

    with pytest.raises(ExecutionError) as caught:
        raise_mapped_client_error(
            error,
            provider="test-provider",
            model="test-model",
            messages=_ERROR_MESSAGES,
        )

    assert caught.value.metadata is not None
    assert caught.value.metadata.retry_after is None
