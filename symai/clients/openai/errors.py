from symai.clients import errors as integration_errors
from symai.clients import http_errors
from symai.clients.openai.transport import ResponseMetadata


class Error(integration_errors.IntegrationError):
    """Base class for all typed errors raised by the OpenAI client."""

    integration = "openai"


class APIError(http_errors.APIError, Error):
    def __init__(self, metadata: ResponseMetadata, body: str, message: str | None = None) -> None:
        self.metadata = metadata
        self.body = body
        super().__init__(message or f"OpenAI API error {metadata.status_code}")

    @property
    def status_code(self) -> int:
        return self.metadata.status_code

    @property
    def request_id(self) -> str | None:
        return self.metadata.request_id


class AuthError(APIError, http_errors.AuthError):
    """Raised when OpenAI rejects the request as unauthenticated."""


class RateLimitError(APIError, http_errors.RateLimitError):
    @property
    def retry_after(self) -> float | None:
        return self.metadata.retry_after


class ResponseError(integration_errors.ResponseError, Error):
    def __init__(self, message: str, *, metadata: ResponseMetadata, body: str) -> None:
        self.metadata = metadata
        self.body = body
        super().__init__(message)


class TransportError(integration_errors.TransportError, Error):
    def __init__(self, message: str) -> None:
        self.metadata: None = None
        super().__init__(message)
