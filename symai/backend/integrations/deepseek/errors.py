from symai.backend.integrations import errors as integration_errors
from symai.backend.integrations import http_errors
from symai.backend.integrations.deepseek.transport import ResponseMetadata


class Error(integration_errors.IntegrationError):
    """Base class for all typed errors raised by the DeepSeek client."""

    integration = "deepseek"


class APIError(http_errors.APIError, Error):
    """Raised for a non-2xx response from the DeepSeek API."""

    def __init__(
        self,
        metadata: ResponseMetadata,
        body: str,
        message: str | None = None,
    ) -> None:
        self.metadata = metadata
        self.body = body
        super().__init__(message or f"DeepSeek API error {metadata.status_code}")

    @property
    def status_code(self) -> int:
        return self.metadata.status_code

    @property
    def request_id(self) -> str | None:
        return self.metadata.request_id


class AuthError(APIError, http_errors.AuthError):
    """Raised when the DeepSeek API rejects the request as unauthenticated."""


class RateLimitError(APIError, http_errors.RateLimitError):
    """Raised when the DeepSeek API reports that the request was rate limited."""

    @property
    def retry_after(self) -> float | None:
        return self.metadata.retry_after


class ResponseError(integration_errors.ResponseError, Error):
    """Raised when a 2xx response body cannot be decoded or validated."""

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


class TransportError(integration_errors.TransportError, Error):
    """Raised when the request fails before a valid HTTP response is received."""

    def __init__(self, message: str) -> None:
        self.metadata: None = None
        super().__init__(message)
