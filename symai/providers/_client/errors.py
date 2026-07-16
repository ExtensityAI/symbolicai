from symai.providers._client.transport import ResponseMetadata


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
        self.body = body
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
        self.body = body
        super().__init__(
            message or f"{self.provider_display_name} API error {metadata.status_code}"
        )


class AuthError(APIError):
    """Raised when an HTTP provider rejects the request as unauthenticated."""


class RateLimitError(APIError):
    """Raised when an HTTP provider rate-limits the request."""
