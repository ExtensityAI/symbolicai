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
    """Synchronous caller-owned client for the Cerebras chat endpoint."""

    def __init__(self, *, api_key: SecretStr, http_client: httpx.Client) -> None:
        authorization = authorization_header(api_key)
        self._http_client = http_client
        self._headers = {"authorization": authorization}

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
