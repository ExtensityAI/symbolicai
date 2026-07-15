import traceback

import pytest
from pydantic import SecretStr

from symai.providers._client.headers import (
    authorization_header,
    parse_optional_float,
    parse_optional_int,
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

    assert failure[:3] == ((), "", "ValueError\n")
    assert failure == _serialize_authorization_failure("\r")


def test_authorization_header_rejects_plaintext_api_key():
    with pytest.raises(TypeError) as exc_info:
        authorization_header("test-key")  # pyright: ignore[reportArgumentType]

    assert exc_info.value.args == ()
    assert str(exc_info.value) == ""


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
