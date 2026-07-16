from symai.providers._client import errors as client_errors


class Error(client_errors.ClientError):
    """Base class for all typed errors raised by the DeepSeek client."""


class APIError(client_errors.APIError, Error):
    """Raised for a non-2xx response from the DeepSeek API."""

    provider_display_name = "DeepSeek"


class AuthError(APIError, client_errors.AuthError):
    """Raised when the DeepSeek API rejects the request as unauthenticated."""


class RateLimitError(APIError, client_errors.RateLimitError):
    """Raised when the DeepSeek API reports that the request was rate limited."""


class ResponseError(client_errors.ResponseError, Error):
    """Raised when a 2xx response body cannot be decoded or validated."""


class TransportError(client_errors.TransportError, Error):
    """Raised when the request fails before a valid HTTP response is received."""
