from symai.providers._client import errors as client_errors
from symai.providers.openai.client.transport import ResponseMetadata


class Error(client_errors.ClientError):
    """Base class for all typed errors raised by the OpenAI client."""

    provider = "openai"


class APIError(client_errors.APIError, Error):
    def __init__(self, metadata: ResponseMetadata, body: str, message: str | None = None) -> None:
        self.metadata = metadata
        self.body = body
        super().__init__(message or f"OpenAI API error {metadata.status_code}")


class AuthError(APIError, client_errors.AuthError):
    """Raised when OpenAI rejects the request as unauthenticated."""


class RateLimitError(APIError, client_errors.RateLimitError):
    """Raised when OpenAI rate-limits the request."""


class ResponseError(client_errors.ResponseError, Error):
    def __init__(self, message: str, *, metadata: ResponseMetadata, body: str) -> None:
        self.metadata = metadata
        self.body = body
        super().__init__(message)


class TransportError(client_errors.TransportError, Error):
    def __init__(self, message: str) -> None:
        self.metadata: None = None
        super().__init__(message)
