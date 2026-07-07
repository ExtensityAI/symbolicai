class CerebrasError(Exception):
    """Base class for all typed errors raised by `CerebrasClient`."""


class CerebrasAuthError(CerebrasError):
    """Raised when the Cerebras API rejects the request as unauthenticated (HTTP 401)."""


class CerebrasRateLimitError(CerebrasError):
    """Raised when the Cerebras API reports the request was rate limited (HTTP 429)."""


class CerebrasAPIError(CerebrasError):
    """Raised for any other non-2xx response from the Cerebras API.

    Carries the raw `status_code` and response `body` so callers can inspect the
    failure without re-parsing the underlying HTTP response.
    """

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Cerebras API error {status_code}: {body}")


class CerebrasResponseError(CerebrasError):
    """Raised when a 2xx response body fails to validate against `ChatResponse`."""
