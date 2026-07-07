from symai.backend.providers import errors as provider_errors


class Error(provider_errors.ProviderError):
    """Base class for all typed errors raised by `cerebras.Client`."""

    provider = "cerebras"


class APIError(provider_errors.APIError, Error):
    """Raised for a non-2xx response from the Cerebras API.

    Carries the raw `status_code` and response `body` so callers can inspect the
    failure without depending on the underlying HTTP client.
    """

    def __init__(
        self,
        status_code: int,
        body: str,
        message: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(message or f"Cerebras API error {status_code}")


class AuthError(APIError, provider_errors.AuthError):
    """Raised when the Cerebras API rejects the request as unauthenticated.

    Usually corresponds to HTTP 401.
    """


class RateLimitError(APIError, provider_errors.RateLimitError):
    """Raised when the Cerebras API reports that the request was rate limited.

    Usually corresponds to HTTP 429.
    """


class ResponseError(provider_errors.ResponseError, Error):
    """Raised when a 2xx response body cannot be decoded or validated.

    Carries the raw response `body` so callers can inspect the failure without
    depending on the underlying HTTP client.
    """

    def __init__(self, message: str, *, body: str) -> None:
        self.body = body
        super().__init__(message)


class TransportError(provider_errors.TransportError, Error):
    """Raised when the request fails before a valid HTTP response is received.

    This includes connection failures, DNS failures, TLS errors, and timeouts.
    The original transport exception should be preserved with exception chaining.
    """
