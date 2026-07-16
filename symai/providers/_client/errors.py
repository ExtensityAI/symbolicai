import json
from dataclasses import dataclass

from symai.providers._client.transport import ResponseMetadata

# Provider error bodies are unbounded and may echo request content back at us. Keep a
# short diagnostic slice of the body, bound every extracted field, and refuse to parse
# anything implausibly large. Each extracted field is bounded rather than the parse being
# skipped, so a huge body still yields its short `code`/`param` instead of nothing.
_MAX_BODY_CHARS = 2_048
_MAX_FIELD_CHARS = 512
_MAX_PARSE_CHARS = 1_048_576


@dataclass(frozen=True, slots=True)
class ProviderErrorDetails:
    """The safe, bounded fields of an OpenAI-compatible provider error envelope.

    `message` is the provider's own text. It is kept as data rather than folded into an
    exception string, because a provider may quote the offending request back at us.
    """

    code: str | None = None
    type: str | None = None
    param: str | None = None
    message: str | None = None


def bounded_body(body: str) -> str:
    """Truncate a provider body to a size safe to retain on an exception."""
    if len(body) <= _MAX_BODY_CHARS:
        return body

    return f"{body[:_MAX_BODY_CHARS]}… ({len(body)} chars total)"


def parse_error_details(body: str) -> ProviderErrorDetails:
    """Best-effort parse of an OpenAI-compatible `{"error": {...}}` envelope.

    Never raises: an unparseable body simply yields empty details, because error
    reporting must not fail on the shape of an error.
    """
    if not body or len(body) > _MAX_PARSE_CHARS:
        return ProviderErrorDetails()

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ProviderErrorDetails()

    if not isinstance(payload, dict):
        return ProviderErrorDetails()

    envelope = payload.get("error", payload)
    if not isinstance(envelope, dict):
        return ProviderErrorDetails()

    return ProviderErrorDetails(
        code=_text_field(envelope, "code"),
        type=_text_field(envelope, "type"),
        param=_text_field(envelope, "param"),
        message=_text_field(envelope, "message"),
    )


def _text_field(envelope: dict[str, object], name: str) -> str | None:
    value = envelope.get(name)
    if isinstance(value, str) and value:
        return _bounded_field(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)

    return None


def _bounded_field(value: str) -> str:
    if len(value) <= _MAX_FIELD_CHARS:
        return value

    return f"{value[:_MAX_FIELD_CHARS]}…"


class ClientError(Exception):
    """Base class for every error raised by a provider client."""


class TransportError(ClientError):
    """Raised when the provider could not be reached at all.

    Connection, DNS, TLS, and process-spawn failures, and timeouts — anything that
    went wrong before a well-formed response existed.
    """

    def __init__(self, message: str) -> None:
        self.metadata: None = None
        super().__init__(message)


class ResponseError(ClientError):
    """Raised when the provider response could not be decoded or validated."""

    def __init__(
        self,
        message: str,
        *,
        metadata: ResponseMetadata,
        body: str,
    ) -> None:
        self.metadata = metadata
        self.body = bounded_body(body)
        self.details = parse_error_details(body)
        super().__init__(message)


class APIError(ClientError):
    """Raised when an HTTP provider returns a non-success response."""

    provider_display_name = "Provider"

    def __init__(
        self,
        metadata: ResponseMetadata,
        body: str,
        message: str | None = None,
    ) -> None:
        self.metadata = metadata
        self.body = bounded_body(body)
        self.details = parse_error_details(body)
        super().__init__(
            message or f"{self.provider_display_name} API error {metadata.status_code}"
        )


class AuthError(APIError):
    """Raised when an HTTP provider rejects the request as unauthenticated."""


class RateLimitError(APIError):
    """Raised when an HTTP provider rate-limits the request."""
