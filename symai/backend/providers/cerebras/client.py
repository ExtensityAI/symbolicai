import re

import httpx
from pydantic import ValidationError

from symai.backend.providers.cerebras.errors import (
    CerebrasAPIError,
    CerebrasAuthError,
    CerebrasConnectionError,
    CerebrasRateLimitError,
    CerebrasResponseError,
)
from symai.backend.providers.cerebras.request import ChatRequest
from symai.backend.providers.cerebras.response import ChatResponse

CHAT_COMPLETIONS_PATH = "/chat/completions"

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_thinking(content: str) -> tuple[str | None, str]:
    """Split a `<think>...</think>` reasoning block out of raw model content.

    Matches the first `<think>...</think>` block, DOTALL so it spans newlines.
    Returns `(thinking, cleaned_content)` with both trimmed; `thinking` is `None`
    if the block is absent or empty. `content` is returned unchanged (not
    trimmed) when no block is found. Pure: `CerebrasClient` never applies this
    automatically, so callers can access raw content.
    """
    match = _THINK_BLOCK.search(content)
    if match is None:
        return None, content

    thinking = match.group(1).strip()
    cleaned = (content[: match.start()] + content[match.end() :]).strip()
    return thinking or None, cleaned


class CerebrasClient:
    """Thin httpx-backed client for the Cerebras chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.cerebras.ai/v1",
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._owns_client = http_client is None

        if http_client is not None:
            self._http_client = http_client
        else:
            # `max_retries` covers connection-level/transport retries only (e.g. connect
            # failures). Retrying on non-2xx status codes with backoff is a policy concern
            # deferred to the future adapter layer.
            transport = httpx.HTTPTransport(retries=max_retries)
            self._http_client = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        """Close the underlying `httpx.Client`, but only if this client created it.

        An injected `http_client` is owned by the caller and is left open.
        """
        if self._owns_client:
            self._http_client.close()

    def __enter__(self) -> "CerebrasClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def create(self, request: ChatRequest) -> ChatResponse:
        """POST a chat completion request and return the typed response.

        Raises:
            CerebrasConnectionError: the request never reached a response (connect
                failure, DNS, TLS, or timeout).
            CerebrasAuthError: the API rejected the request as unauthenticated (401).
            CerebrasRateLimitError: the API rate-limited the request (429).
            CerebrasAPIError: any other non-2xx response.
            CerebrasResponseError: the 2xx body failed to decode as JSON or schema validation.
        """
        body = request.model_dump(mode="json", by_alias=True, exclude_none=True)

        try:
            response = self._http_client.post(
                f"{self._base_url}{CHAT_COMPLETIONS_PATH}",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.HTTPError as e:
            msg = f"Cerebras request failed: {e}"
            raise CerebrasConnectionError(msg) from e

        if response.status_code == httpx.codes.UNAUTHORIZED:
            msg = f"Cerebras API rejected credentials: {response.text}"
            raise CerebrasAuthError(msg)

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            msg = f"Cerebras API rate limit exceeded: {response.text}"
            raise CerebrasRateLimitError(msg)

        if not response.is_success:
            raise CerebrasAPIError(response.status_code, response.text)

        try:
            payload = response.json()
            return ChatResponse.model_validate(payload)
        except (ValueError, ValidationError) as e:
            msg = f"Cerebras response failed to decode or validate: {e}"
            raise CerebrasResponseError(msg, body=response.text) from e
