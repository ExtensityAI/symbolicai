import pytest

from symai.clients import errors as client_errors
from symai.clients import http_errors
from symai.clients.cerebras import errors as cerebras_errors
from symai.clients.cerebras.transport import RateLimitState, ResponseMetadata


def _metadata(
    *,
    status_code: int = 500,
    request_id: str | None = "req-1",
    retry_after: float | None = None,
) -> ResponseMetadata:
    return ResponseMetadata(
        status_code=status_code,
        request_id=request_id,
        retry_after=retry_after,
        rate_limit=RateLimitState(),
    )


def _provider_errors():
    return (
        cerebras_errors.APIError(_metadata(), "api"),
        cerebras_errors.AuthError(_metadata(status_code=401), "auth"),
        cerebras_errors.RateLimitError(_metadata(status_code=429), "rate limit"),
        cerebras_errors.ResponseError(
            "response",
            metadata=_metadata(status_code=200),
            body="{}",
        ),
        cerebras_errors.TransportError("transport"),
    )


@pytest.mark.parametrize(
    ("error", "catch_type"),
    [
        (cerebras_errors.APIError(_metadata(), "api"), http_errors.APIError),
        (
            cerebras_errors.AuthError(_metadata(status_code=401), "auth"),
            http_errors.AuthError,
        ),
        (
            cerebras_errors.AuthError(_metadata(status_code=401), "auth"),
            cerebras_errors.APIError,
        ),
        (
            cerebras_errors.RateLimitError(
                _metadata(status_code=429),
                "rate limit",
            ),
            http_errors.RateLimitError,
        ),
        (
            cerebras_errors.RateLimitError(
                _metadata(status_code=429),
                "rate limit",
            ),
            cerebras_errors.APIError,
        ),
        (
            cerebras_errors.ResponseError(
                "response",
                metadata=_metadata(status_code=200),
                body="{}",
            ),
            client_errors.ResponseError,
        ),
        (
            cerebras_errors.TransportError("transport"),
            client_errors.TransportError,
        ),
    ],
)
def test_cerebras_errors_are_caught_by_shared_lattices(
    error: cerebras_errors.Error,
    catch_type: type[Exception],
):
    with pytest.raises(catch_type):
        raise error


@pytest.mark.parametrize(
    "catch_type",
    [cerebras_errors.Error, client_errors.ClientError],
)
@pytest.mark.parametrize("error", _provider_errors())
def test_all_cerebras_errors_are_caught_by_shared_bases(
    error: cerebras_errors.Error,
    catch_type: type[Exception],
):
    with pytest.raises(catch_type):
        raise error


def test_api_error_retains_metadata_and_body():
    metadata = _metadata(status_code=500, request_id="req-1")
    error = cerebras_errors.APIError(metadata, "server error")

    assert error.provider == "cerebras"
    assert error.metadata is metadata
    assert error.metadata.status_code == 500
    assert error.metadata.request_id == "req-1"
    assert error.body == "server error"


def test_rate_limit_error_retains_retry_metadata():
    metadata = _metadata(status_code=429, retry_after=2.5)
    error = cerebras_errors.RateLimitError(metadata, "slow")

    assert error.metadata.retry_after == 2.5
    assert error.metadata.status_code == 429


def test_response_error_retains_metadata_and_body():
    metadata = _metadata(status_code=200)
    error = cerebras_errors.ResponseError(
        "bad response",
        metadata=metadata,
        body="{bad json}",
    )

    assert error.metadata is metadata
    assert error.body == "{bad json}"


def test_transport_error_defaults_to_no_metadata():
    error = cerebras_errors.TransportError("network failure")
    assert error.metadata is None
