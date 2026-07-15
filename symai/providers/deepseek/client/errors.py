from symai.providers._client import errors as client_errors
from symai.providers.deepseek.client.transport import ResponseMetadata


class Error(client_errors.ClientError):
    """Base class for all typed errors raised by the DeepSeek client."""

    provider = "deepseek"


class APIError(client_errors.APIError, Error):
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


class AuthError(APIError, client_errors.AuthError):
    """Raised when the DeepSeek API rejects the request as unauthenticated."""


class RateLimitError(APIError, client_errors.RateLimitError):
    """Raised when the DeepSeek API reports that the request was rate limited."""


class ResponseError(client_errors.ResponseError, Error):
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


class TransportError(client_errors.TransportError, Error):
    """Raised when the request fails before a valid HTTP response is received."""

    def __init__(self, message: str) -> None:
        self.metadata: None = None
        super().__init__(message)
