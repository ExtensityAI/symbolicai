import json

import httpx
from pydantic import SecretStr, ValidationError

import symai.providers.cerebras.client.chat as chat
import symai.providers.cerebras.client.errors as errors
from symai.providers._client.headers import authorization_header
from symai.providers.cerebras.client.headers import extract_response_metadata
from symai.providers.cerebras.client.transport import APIResponse, ResponseMetadata

BASE_URL = "https://api.cerebras.ai/v1"




def _raise_for_status(
    response: httpx.Response,
    metadata: ResponseMetadata,
):
    if response.status_code == httpx.codes.UNAUTHORIZED:
        raise errors.AuthError(
            metadata,
            response.text,
            "Cerebras API rejected credentials",
        )
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        raise errors.RateLimitError(
            metadata,
            response.text,
            "Cerebras API rate limit exceeded",
        )
    if not response.is_success:
        raise errors.APIError(metadata, response.text)


def _parse_response(
    response: httpx.Response,
    metadata: ResponseMetadata,
):
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        message = "Cerebras response was not valid JSON"
        raise errors.ResponseError(
            message,
            metadata=metadata,
            body=response.text,
        ) from exc

    try:
        return chat.ChatCompletion.model_validate(payload)
    except ValidationError as exc:
        message = "Cerebras response did not match the expected schema"
        raise errors.ResponseError(
            message,
            metadata=metadata,
            body=response.text,
        ) from exc


class Client:
    """Synchronous owner of a Cerebras HTTP connection pool."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout | float = 5.0,
        connect_retries: int = 0,
    ) -> None:
        authorization = authorization_header(api_key)
        owned_transport = transport
        if owned_transport is None:
            owned_transport = httpx.HTTPTransport(retries=connect_retries)
        elif connect_retries:
            msg = "connect_retries cannot be combined with an injected transport"
            raise ValueError(msg)

        try:
            http_client = httpx.Client(timeout=timeout, transport=owned_transport)
        except BaseException as error:
            try:
                owned_transport.close()
            except BaseException as cleanup_error:
                error.add_note(f"Client construction cleanup failed: {cleanup_error!r}")
            raise

        self._http_client = http_client
        self._headers = {"authorization": authorization}
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._http_client.close()

    def create_chat_completion(
        self,
        request: chat.CreateChatCompletionRequest,
    ) -> APIResponse[chat.ChatCompletion]:
        """Execute one non-streaming chat completion request."""

        try:
            response = self._http_client.post(
                f"{BASE_URL}{chat.PATH}",
                json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
                headers=self._headers,
            )
        except httpx.RequestError as exc:
            message = "Cerebras request failed before receiving a valid response"
            raise errors.TransportError(message) from exc

        metadata = extract_response_metadata(response)
        _raise_for_status(response, metadata)
        return APIResponse(data=_parse_response(response, metadata), metadata=metadata)
