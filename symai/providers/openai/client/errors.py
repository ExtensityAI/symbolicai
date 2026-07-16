from symai.providers._client import errors as client_errors


class Error(client_errors.ClientError):
    """Base class for all typed errors raised by the OpenAI client."""


class APIError(client_errors.APIError, Error):
    provider_display_name = "OpenAI"


class AuthError(APIError, client_errors.AuthError):
    """Raised when OpenAI rejects the request as unauthenticated."""


class RateLimitError(APIError, client_errors.RateLimitError):
    """Raised when OpenAI rate-limits the request."""


class ResponseError(client_errors.ResponseError, Error):
    """Raised when an OpenAI response cannot be decoded or validated."""


class TransportError(client_errors.TransportError, Error):
    """Raised when an OpenAI request fails before receiving a response."""
