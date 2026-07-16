import httpx
from pydantic import SecretStr

from symai.providers._http.response import HttpMetadata

_UNSAFE_API_KEY_MESSAGE = (
    "api_key must be nonempty and free of surrounding whitespace and control characters"
)
_PLAINTEXT_API_KEY_MESSAGE = "api_key must be a SecretStr"


def authorization_header(api_key: SecretStr) -> str:
    """Build the Authorization header, rejecting a credential unsafe to put on the wire.

    Every rejection raises the same static message. It is deliberately constant rather
    than naming the rule that failed: which check a key trips is itself a property of the
    secret, and the message must never be derived from the credential.

    Raises:
        TypeError: if `api_key` is not a SecretStr.
        ValueError: if the credential is empty, surrounded by whitespace, or contains any
            character that is not visible ASCII.
    """
    if not isinstance(api_key, SecretStr):
        raise TypeError(_PLAINTEXT_API_KEY_MESSAGE)

    value = api_key.get_secret_value()
    invalid = not value or value[0].isspace() or value[-1].isspace()
    if not invalid:
        for character in value:
            code_point = ord(character)
            # A header value is visible ASCII on the wire. Anything else — a control
            # character, or a non-ASCII character from a mis-decoded key — must fail here:
            # httpx would otherwise raise UnicodeEncodeError while encoding the header,
            # carrying the credential in the exception's arguments.
            if code_point < 0x20 or code_point >= 0x7F:
                invalid = True
                break

    if invalid:
        raise ValueError(_UNSAFE_API_KEY_MESSAGE)

    return f"Bearer {value}"


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def extract_response_metadata(response: httpx.Response) -> HttpMetadata:
    return HttpMetadata(
        status_code=response.status_code,
        request_id=response.headers.get("x-request-id"),
        retry_after=parse_optional_float(response.headers.get("retry-after")),
    )
