"""Error categories shared by HTTP provider clients."""

from symai.clients.errors import ClientError


class APIError(ClientError):
    """Raised when an HTTP provider returns a non-success response."""


class AuthError(APIError):
    """Raised when an HTTP provider rejects the request as unauthenticated."""


class RateLimitError(APIError):
    """Raised when an HTTP provider rate-limits the request."""
