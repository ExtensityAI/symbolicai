import json

import httpx
from pydantic import SecretStr, ValidationError

import symai.providers.deepseek.client.chat as chat
import symai.providers.deepseek.client.errors as errors
from symai.providers._client.headers import authorization_header
from symai.providers.deepseek.client.headers import extract_response_metadata
from symai.providers.deepseek.client.transport import APIResponse, ResponseMetadata

BASE_URL = "https://api.deepseek.com"




def _raise_for_status(response: httpx.Response, metadata: ResponseMetadata):
    if response.status_code == httpx.codes.UNAUTHORIZED:
        raise errors.AuthError(
            metadata,
            response.text,
            "DeepSeek API rejected credentials",
        )
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        raise errors.RateLimitError(
            metadata,
            response.text,
            "DeepSeek API rate limit exceeded",
        )
    if not response.is_success:
        raise errors.APIError(metadata, response.text)


def _parse_response(response: httpx.Response, metadata: ResponseMetadata):
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        message = "DeepSeek response was not valid JSON"
        raise errors.ResponseError(
            message,
            metadata=metadata,
            body=response.text,
        ) from exc

    try:
        return chat.ChatCompletion.model_validate(payload)
    except ValidationError as exc:
        message = "DeepSeek response did not match the expected schema"
        raise errors.ResponseError(
            message,
            metadata=metadata,
            body=response.text,
        ) from exc


class Client:
    """Synchronous caller-owned client for the DeepSeek chat endpoint."""

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
                json=request.model_dump(mode="json", exclude_none=True),
                headers=self._headers,
            )
        except httpx.RequestError as exc:
            message = "DeepSeek request failed before receiving a valid response"
            raise errors.TransportError(message) from exc

        metadata = extract_response_metadata(response)
        _raise_for_status(response, metadata)
        return APIResponse(data=_parse_response(response, metadata), metadata=metadata)
