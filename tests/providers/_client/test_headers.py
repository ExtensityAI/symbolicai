import traceback

import httpx
import pytest
from pydantic import SecretStr

from symai.providers._client.headers import (
    _UNSAFE_API_KEY_MESSAGE,
    authorization_header,
    extract_response_metadata,
    parse_optional_float,
    parse_optional_int,
)
from symai.providers.cerebras.client.headers import (
    extract_response_metadata as extract_cerebras_metadata,
)


def _serialize_authorization_failure(api_key: str):
    try:
        authorization_header(SecretStr(api_key))
    except ValueError as error:
        return (
            error.args,
            str(error),
            "".join(traceback.format_exception_only(type(error), error)),
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        )

    msg = "authorization_header accepted an unsafe credential"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    "api_key",
    [
        "",
        "secret\rvalue",
        "secret\nvalue",
        "secret\x00value",
        "secret\x01value",
        "secret\x1fvalue",
        "secret\x7fvalue",
        " secret",
        "secret ",
        "api_key ",
        " ",
        "\u2003secret",
        "secret\u2003",
        "ValueError\n",
        "TypeError\n",
    ],
    ids=[
        "empty",
        "cr",
        "lf",
        "nul",
        "c0-start",
        "c0-end",
        "del",
        "leading-space",
        "trailing-space",
        "message-collision",
        "whitespace-only",
        "leading-unicode-space",
        "trailing-unicode-space",
        "value-error-traceback-text",
        "type-error-traceback-text",
    ],
)
def test_authorization_header_rejects_unsafe_api_key_without_disclosure(api_key: str):
    failure = _serialize_authorization_failure(api_key)

    # The message is a constant, so it cannot be derived from the credential, and every
    # invalid credential must fail identically: which rule a key trips is itself a
    # property of the secret.
    assert failure[0] == (_UNSAFE_API_KEY_MESSAGE,)
    assert failure[1] == _UNSAFE_API_KEY_MESSAGE
    assert failure == _serialize_authorization_failure("\r")


def test_authorization_header_rejects_plaintext_api_key():
    with pytest.raises(TypeError) as exc_info:
        authorization_header("test-key")  # pyright: ignore[reportArgumentType]

    assert exc_info.value.args == ("api_key must be a SecretStr",)


@pytest.mark.parametrize(
    "api_key",
    [
        "test-key",
        "sk_live.123_ABC+/=:@!#$%^&*?~",
        "key with internal space",
    ],
)
def test_authorization_header_preserves_valid_api_key(api_key: str):
    assert authorization_header(SecretStr(api_key)) == f"Bearer {api_key}"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("invalid", None), ("2.5", 2.5)],
)
def test_parse_optional_float(value, expected):
    assert parse_optional_float(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("invalid", None), ("42", 42)],
)
def test_parse_optional_int(value, expected):
    assert parse_optional_int(value) == expected


def test_shared_metadata_extractor_reads_status_request_id_and_retry_after():
    response = httpx.Response(
        429,
        headers={"x-request-id": "req-1", "retry-after": "1.5"},
    )

    metadata = extract_response_metadata(response)

    assert metadata.status_code == 429
    assert metadata.request_id == "req-1"
    assert metadata.retry_after == 1.5


def test_shared_metadata_extractor_tolerates_absent_and_unparseable_headers():
    metadata = extract_response_metadata(httpx.Response(200, headers={"retry-after": "soon"}))

    assert metadata.status_code == 200
    assert metadata.request_id is None
    assert metadata.retry_after is None


def test_cerebras_metadata_extractor_adds_rate_limit_headers():
    response = httpx.Response(
        429,
        headers={
            "x-request-id": "req-1",
            "retry-after": "1.5",
            "x-ratelimit-limit-requests-day": "100",
            "x-ratelimit-limit-tokens-minute": "1000",
            "x-ratelimit-remaining-requests-day": "99",
            "x-ratelimit-remaining-tokens-minute": "900",
            "x-ratelimit-reset-requests-day": "30.5",
            "x-ratelimit-reset-tokens-minute": "5.5",
        },
    )

    metadata = extract_cerebras_metadata(response)

    assert metadata.status_code == 429
    assert metadata.request_id == "req-1"
    assert metadata.retry_after == 1.5
    assert metadata.rate_limit.model_dump() == {
        "limit_requests_day": 100,
        "limit_tokens_minute": 1000,
        "remaining_requests_day": 99,
        "remaining_tokens_minute": 900,
        "reset_requests_day": 30.5,
        "reset_tokens_minute": 5.5,
    }
