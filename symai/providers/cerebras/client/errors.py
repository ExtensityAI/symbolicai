from symai.providers._http import errors as client_errors


class Error(client_errors.ClientError):
    """Base class for all typed errors raised by the Cerebras client."""


class APIError(client_errors.APIError, Error):
    """Raised for a non-2xx response from the Cerebras API.

    The exact response body and transport metadata remain available without
    exposing the underlying HTTP client.
    """

    provider_display_name = "Cerebras"


class AuthError(APIError, client_errors.AuthError):
    """Raised when the Cerebras API rejects the request as unauthenticated."""


class RateLimitError(APIError, client_errors.RateLimitError):
    """Raised when the Cerebras API reports that the request was rate limited."""


class ResponseError(client_errors.ResponseError, Error):
    """Raised when a 2xx response body cannot be decoded or validated."""


class TransportError(client_errors.TransportError, Error):
    """Raised when the request fails before a valid HTTP response is received."""
