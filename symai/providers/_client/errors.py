class ClientError(Exception):
    """Base class for every error raised by a provider client."""


class TransportError(ClientError):
    """Raised when the provider could not be reached at all.

    Connection, DNS, TLS, and process-spawn failures, and timeouts — anything that
    went wrong before a well-formed response existed.
    """


class ResponseError(ClientError):
    """Raised when the provider response could not be decoded or validated."""


class APIError(ClientError):
    """Raised when an HTTP provider returns a non-success response."""


class AuthError(APIError):
    """Raised when an HTTP provider rejects the request as unauthenticated."""


class RateLimitError(APIError):
    """Raised when an HTTP provider rate-limits the request."""
