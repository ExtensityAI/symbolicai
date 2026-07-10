import httpx
from pydantic import ValidationError

from symai.backend.integrations.cerebras.client import errors
from symai.backend.integrations.cerebras.client.chat import ChatRequest, ChatResponse

CHAT_COMPLETIONS_PATH = "/chat/completions"
BASE_URL = "https://api.cerebras.ai/v1"
REQUEST_ID_HEADER = "x-request-id"
RETRY_AFTER_HEADER = "retry-after"


def _retry_after(response: httpx.Response) -> float | None:
    """The API's own retry instruction, in seconds, when it sends one.

    Only the delta-seconds form of `Retry-After` is understood; an HTTP-date value
    yields None rather than a guess. The client surfaces the instruction and never
    acts on it, because retrying is the caller's policy.
    """

    raw = response.headers.get(RETRY_AFTER_HEADER)

    if raw is None:
        return None

    try:
        return float(raw)
    except ValueError:
        return None


def _request_body(request: ChatRequest) -> dict[str, object]:
    return request.model_dump(mode="json", by_alias=True, exclude_none=True)


class Client:
    """Thin httpx-backed client for the Cerebras chat completions endpoint.

    The caller supplies and owns the `httpx.Client`. This class never closes it, and
    every transport policy — timeout, connection retries, proxies, limits — is the
    caller's to set. Note that httpx defaults to a 5 second timeout, far too short
    for chat completions, so set one explicitly.
    """

    def __init__(self, api_key: str, *, http_client: httpx.Client) -> None:
        if not api_key:
            msg = "api_key must not be empty"
            raise ValueError(msg)

        self._api_key = api_key
        self._http_client = http_client
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    def create(self, request: ChatRequest) -> ChatResponse:
        """POST a chat completion request and return the typed response.

        Raises:
            errors.TransportError: the request failed before a valid HTTP response
                was received.
            errors.AuthError: the API rejected the request as unauthenticated.
            errors.RateLimitError: the API rate-limited the request.
            errors.APIError: the API returned another non-2xx response.
            errors.ResponseError: the 2xx response body failed to decode or validate.
        """

        try:
            response = self._http_client.post(
                f"{BASE_URL}{CHAT_COMPLETIONS_PATH}",
                json=_request_body(request),
                headers=self._headers,
            )
        except httpx.RequestError as exc:
            msg = "Cerebras request failed before receiving a valid response"
            raise errors.TransportError(msg) from exc

        request_id = response.headers.get(REQUEST_ID_HEADER)

        if response.status_code == httpx.codes.UNAUTHORIZED:
            raise errors.AuthError(
                response.status_code,
                response.text,
                "Cerebras API rejected credentials",
                request_id=request_id,
            )

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise errors.RateLimitError(
                response.status_code,
                response.text,
                "Cerebras API rate limit exceeded",
                request_id=request_id,
                retry_after=_retry_after(response),
            )

        if not response.is_success:
            raise errors.APIError(response.status_code, response.text, request_id=request_id)

        try:
            payload = response.json()
        except ValueError as exc:
            msg = "Cerebras API returned a response that was not valid JSON"
            raise errors.ResponseError(
                msg,
                body=response.text,
            ) from exc

        try:
            return ChatResponse.model_validate(payload)
        except ValidationError as exc:
            msg = "Cerebras API returned a response that did not match the expected schema"
            raise errors.ResponseError(
                msg,
                body=response.text,
            ) from exc
