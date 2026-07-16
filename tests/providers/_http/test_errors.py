import json

import pytest

from symai.providers._http import errors as client_errors
from symai.providers.cerebras.client import errors as cerebras_errors
from symai.providers.cerebras.client.response import HttpMetadata, RateLimitState


def _metadata(
    *,
    status_code: int = 500,
    request_id: str | None = "req-1",
    retry_after: float | None = None,
) -> HttpMetadata:
    return HttpMetadata(
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
        (cerebras_errors.APIError(_metadata(), "api"), client_errors.APIError),
        (
            cerebras_errors.AuthError(_metadata(status_code=401), "auth"),
            client_errors.AuthError,
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
            client_errors.RateLimitError,
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


def test_error_body_is_bounded_but_details_survive_a_huge_payload():
    body = json.dumps({"error": {"message": "x" * 500_000, "code": "context_length_exceeded"}})
    error = cerebras_errors.APIError(_metadata(status_code=400), body)

    assert len(body) > 500_000
    assert len(error.body) < 2_200
    assert error.body.endswith(f"({len(body)} chars total)")
    assert error.details.code == "context_length_exceeded"
    assert error.details.message is not None
    assert len(error.details.message) < 600


def test_error_details_parse_the_openai_compatible_envelope():
    body = json.dumps(
        {
            "error": {
                "message": "Unsupported value: 'temperature'",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": "unsupported_value",
            }
        }
    )
    error = cerebras_errors.APIError(_metadata(status_code=400), body)

    assert error.details.message == "Unsupported value: 'temperature'"
    assert error.details.type == "invalid_request_error"
    assert error.details.param == "temperature"
    assert error.details.code == "unsupported_value"


@pytest.mark.parametrize("body", ["", "not json", "[]", '{"error": "flat"}', "null"])
def test_error_details_never_raise_on_an_unparseable_body(body: str):
    error = cerebras_errors.APIError(_metadata(status_code=500), body)

    assert error.details.code is None
    assert error.details.message is None


def test_provider_body_never_reaches_the_exception_message():
    body = json.dumps({"error": {"message": "SECRET PROMPT ECHO", "code": "bad"}})
    error = cerebras_errors.APIError(_metadata(status_code=400), body)

    assert "SECRET PROMPT ECHO" not in str(error)
    assert error.details.message == "SECRET PROMPT ECHO"
