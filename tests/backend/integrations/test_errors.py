import pytest

from symai.backend.integrations import errors as integration_errors
from symai.backend.integrations import http_errors
from symai.backend.integrations.cerebras.client import errors as cerebras_errors


def test_universal_lattice_is_free_of_http_semantics():
    # A non-HTTP integration (a local Lean4 binding, a subprocess tool) imports only the
    # universal module, and must not inherit status codes, auth, or rate limiting.
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
    # The multiple-inheritance lattice keeps auth/rate-limit catchable as an APIError,
    # at both the shared and the integration-specific level.
    assert issubclass(cerebras_errors.AuthError, cerebras_errors.APIError)
    assert issubclass(cerebras_errors.RateLimitError, cerebras_errors.APIError)
    assert issubclass(cerebras_errors.AuthError, http_errors.APIError)
    assert issubclass(cerebras_errors.RateLimitError, http_errors.APIError)


def test_shared_except_catches_the_integration_specific_error():
    with pytest.raises(http_errors.AuthError):
        raise cerebras_errors.AuthError(401, "nope")

    with pytest.raises(http_errors.APIError):
        raise cerebras_errors.RateLimitError(429, "slow down")

    with pytest.raises(integration_errors.IntegrationError):
        raise cerebras_errors.TransportError("boom")

    with pytest.raises(integration_errors.ResponseError):
        raise cerebras_errors.ResponseError("bad", body="{}")


def test_errors_carry_integration_tag_and_payload():
    assert cerebras_errors.Error.integration == "cerebras"

    api = cerebras_errors.APIError(500, "server error", request_id="req-1")
    assert api.integration == "cerebras"
    assert api.status_code == 500
    assert api.body == "server error"
    assert api.request_id == "req-1"

    resp = cerebras_errors.ResponseError("bad body", body="{not json}")
    assert resp.body == "{not json}"


def test_rate_limit_error_surfaces_the_apis_retry_instruction():
    err = cerebras_errors.RateLimitError(429, "slow", request_id="req-2", retry_after=2.5)

    assert err.retry_after == 2.5
    assert err.request_id == "req-2"
    assert err.status_code == 429


def test_request_id_and_retry_after_default_to_none():
    assert cerebras_errors.APIError(500, "boom").request_id is None
    assert cerebras_errors.RateLimitError(429, "slow").retry_after is None
