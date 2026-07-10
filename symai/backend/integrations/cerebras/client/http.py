from types import TracebackType
from typing import Self

import httpx
from pydantic import ValidationError

from symai.backend.integrations.cerebras.client import errors
from symai.backend.integrations.cerebras.client.chat import ChatRequest, ChatResponse

CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2
REQUEST_ID_HEADER = "x-request-id"
RETRY_AFTER_HEADER = "retry-after"


class Client:
    """Thin httpx-backed client for the Cerebras chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        http_client: httpx.Client | None = None,
    ) -> None:

        if not api_key:
            msg = "api_key must not be empty"
            raise ValueError(msg)

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None

        if http_client is None:
            self._http_client = httpx.Client(
                timeout=DEFAULT_TIMEOUT,
                transport=httpx.HTTPTransport(retries=DEFAULT_RETRIES),
            )
        else:
            self._http_client = http_client

        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    def close(self) -> None:
        """Close the underlying `httpx.Client` if this client owns it."""

        if self._owns_client:
            self._http_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

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
                f"{self._base_url}{CHAT_COMPLETIONS_PATH}",
                json=self._request_body(request),
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
                retry_after=self._retry_after(response),
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

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        """The API's own retry instruction, in seconds, when it sends one.

        Only the delta-seconds form of `Retry-After` is understood; an HTTP-date
        value yields None rather than a guess. The client surfaces the instruction
        and never acts on it, because retrying is the caller's policy.
        """

        raw = response.headers.get(RETRY_AFTER_HEADER)

        if raw is None:
            return None

        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _request_body(request: ChatRequest) -> dict[str, object]:
        return request.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
