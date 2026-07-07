class ProviderError(Exception):
    """Base class for all provider-layer errors."""


class TransportError(ProviderError):
    """Raised when a provider request fails before a valid HTTP response exists."""


class APIError(ProviderError):
    """Raised when a provider returns a non-success API response."""


class AuthError(APIError):
    """Raised when a provider rejects a request as unauthenticated."""


class RateLimitError(APIError):
    """Raised when a provider rate-limits a request."""


class ResponseError(ProviderError):
    """Raised when a provider response cannot be decoded or validated."""
