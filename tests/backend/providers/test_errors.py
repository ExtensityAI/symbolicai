import pytest

from symai.backend.providers import errors as provider_errors
from symai.backend.providers.cerebras import errors as cerebras_errors


def test_cerebras_errors_subclass_the_general_provider_errors():
    assert issubclass(cerebras_errors.Error, provider_errors.ProviderError)
    assert issubclass(cerebras_errors.APIError, provider_errors.APIError)
    assert issubclass(cerebras_errors.AuthError, provider_errors.AuthError)
    assert issubclass(cerebras_errors.RateLimitError, provider_errors.RateLimitError)
    assert issubclass(cerebras_errors.ResponseError, provider_errors.ResponseError)
    assert issubclass(cerebras_errors.TransportError, provider_errors.TransportError)


def test_auth_and_rate_limit_are_also_api_errors_both_levels():
    # The multiple-inheritance lattice keeps auth/rate-limit catchable as an APIError,
    # at both the general and the provider-specific level.
    assert issubclass(cerebras_errors.AuthError, cerebras_errors.APIError)
    assert issubclass(cerebras_errors.RateLimitError, cerebras_errors.APIError)
    assert issubclass(cerebras_errors.AuthError, provider_errors.APIError)
    assert issubclass(cerebras_errors.RateLimitError, provider_errors.APIError)


def test_general_except_catches_the_provider_specific_error():
    with pytest.raises(provider_errors.AuthError):
        raise cerebras_errors.AuthError(401, "nope")

    with pytest.raises(provider_errors.APIError):
        raise cerebras_errors.RateLimitError(429, "slow down")

    with pytest.raises(provider_errors.ProviderError):
        raise cerebras_errors.TransportError("boom")

    with pytest.raises(provider_errors.ResponseError):
        raise cerebras_errors.ResponseError("bad", body="{}")


def test_errors_carry_provider_tag_and_payload():
    assert cerebras_errors.Error.provider == "cerebras"

    api = cerebras_errors.APIError(500, "server error")
    assert api.provider == "cerebras"
    assert api.status_code == 500
    assert api.body == "server error"

    resp = cerebras_errors.ResponseError("bad body", body="{not json}")
    assert resp.body == "{not json}"
