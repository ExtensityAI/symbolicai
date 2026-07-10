import pytest

from symai.backend.integrations import errors as integration_errors
from symai.backend.integrations import http_errors
from symai.backend.integrations.cerebras import errors as cerebras_errors
from symai.backend.integrations.cerebras.response import (
    RateLimitState,
    ResponseMetadata,
)


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


def test_universal_lattice_is_free_of_http_semantics():
    assert not hasattr(integration_errors, "APIError")
    assert not hasattr(integration_errors, "AuthError")
    assert not hasattr(integration_errors, "RateLimitError")


def test_http_errors_are_still_integration_errors():
    assert issubclass(http_errors.APIError, integration_errors.IntegrationError)
    assert issubclass(http_errors.AuthError, http_errors.APIError)
    assert issubclass(http_errors.RateLimitError, http_errors.APIError)


def test_cerebras_errors_subclass_both_shared_lattices():
    assert issubclass(cerebras_errors.Error, integration_errors.IntegrationError)
    assert issubclass(cerebras_errors.TransportError, integration_errors.TransportError)
    assert issubclass(cerebras_errors.ResponseError, integration_errors.ResponseError)
    assert issubclass(cerebras_errors.APIError, http_errors.APIError)
    assert issubclass(cerebras_errors.AuthError, http_errors.AuthError)
    assert issubclass(cerebras_errors.RateLimitError, http_errors.RateLimitError)


def test_auth_and_rate_limit_are_also_api_errors_at_both_levels():
    assert issubclass(cerebras_errors.AuthError, cerebras_errors.APIError)
    assert issubclass(cerebras_errors.RateLimitError, cerebras_errors.APIError)
    assert issubclass(cerebras_errors.AuthError, http_errors.APIError)
    assert issubclass(cerebras_errors.RateLimitError, http_errors.APIError)


def test_shared_except_catches_the_integration_specific_error():
    with pytest.raises(http_errors.AuthError):
        raise cerebras_errors.AuthError(_metadata(status_code=401), "nope")

    with pytest.raises(http_errors.APIError):
        raise cerebras_errors.RateLimitError(_metadata(status_code=429), "slow down")

    transport_message = "boom"
    response_message = "bad"

    with pytest.raises(integration_errors.IntegrationError):
        raise cerebras_errors.TransportError(transport_message)

    with pytest.raises(integration_errors.ResponseError):
        raise cerebras_errors.ResponseError(
            response_message,
            metadata=_metadata(status_code=200),
            body="{}",
        )


def test_api_error_properties_delegate_to_metadata():
    metadata = _metadata(status_code=500, request_id="req-1")
    error = cerebras_errors.APIError(metadata, "server error")

    assert error.integration == "cerebras"
    assert error.metadata is metadata
    assert error.status_code == 500
    assert error.request_id == "req-1"
    assert error.body == "server error"


def test_rate_limit_retry_after_delegates_to_metadata():
    metadata = _metadata(status_code=429, retry_after=2.5)
    error = cerebras_errors.RateLimitError(metadata, "slow")

    assert error.retry_after == 2.5
    assert error.status_code == 429


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


def test_error_compatibility_properties_are_read_only():
    error = cerebras_errors.RateLimitError(
        _metadata(status_code=429, retry_after=1.0),
        "slow",
    )

    with pytest.raises(AttributeError):
        setattr(error, "status_code", 500)
    with pytest.raises(AttributeError):
        setattr(error, "request_id", "other")
    with pytest.raises(AttributeError):
        setattr(error, "retry_after", 3.0)
