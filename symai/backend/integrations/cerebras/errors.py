from symai.backend.integrations import errors as integration_errors
from symai.backend.integrations import http_errors


class Error(integration_errors.IntegrationError):
    """Base class for all typed errors raised by the Cerebras client."""

    integration = "cerebras"


class APIError(http_errors.APIError, Error):
    """Raised for a non-2xx response from the Cerebras API.

    Carries the raw `status_code`, the response `body`, and the API's `request_id`
    when it sends one, so callers can inspect the failure — and quote it to
    support — without depending on the underlying HTTP client.
    """

    def __init__(
        self,
        status_code: int,
        body: str,
        message: str | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.request_id = request_id
        super().__init__(message or f"Cerebras API error {status_code}")


class AuthError(APIError, http_errors.AuthError):
    """Raised when the Cerebras API rejects the request as unauthenticated.

    Usually corresponds to HTTP 401.
    """


class RateLimitError(APIError, http_errors.RateLimitError):
    """Raised when the Cerebras API reports that the request was rate limited.

    Usually corresponds to HTTP 429. `retry_after` carries the API's own retry
    instruction in seconds when it sends one; the client surfaces it but never acts
    on it, because retrying is the caller's policy.
    """

    def __init__(
        self,
        status_code: int,
        body: str,
        message: str | None = None,
        *,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(status_code, body, message, request_id=request_id)


class ResponseError(integration_errors.ResponseError, Error):
    """Raised when a 2xx response body cannot be decoded or validated.

    Carries the raw response `body` so callers can inspect the failure without
    depending on the underlying HTTP client.
    """

    def __init__(self, message: str, *, body: str) -> None:
        self.body = body
        super().__init__(message)


class TransportError(integration_errors.TransportError, Error):
    """Raised when the request fails before a valid HTTP response is received.

    This includes connection failures, DNS failures, TLS errors, and timeouts.
    The original transport exception is preserved with exception chaining.
    """
