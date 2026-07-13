"""Errors specific to HTTP-backed integrations.

Kept out of `integrations.errors` so that integrations which speak no HTTP — a
local Lean4 binding, a subprocess tool — are not forced to inherit status codes,
authentication, or rate limiting. HTTP integrations opt in by importing here.
"""

from symai.clients.errors import IntegrationError


class APIError(IntegrationError):
    """Raised when an HTTP integration returns a non-success response."""


class AuthError(APIError):
    """Raised when an HTTP integration rejects the request as unauthenticated."""


class RateLimitError(APIError):
    """Raised when an HTTP integration rate-limits the request."""
