import json

import httpx
from pydantic import ValidationError

from symai.clients.cerebras import chat, errors
from symai.clients.cerebras.transport import (
    APIResponse,
    RateLimitState,
    ResponseMetadata,
)

BASE_URL = "https://api.cerebras.ai/v1"
REQUEST_ID_HEADER = "x-request-id"
RETRY_AFTER_HEADER = "retry-after"
LIMIT_REQUESTS_DAY_HEADER = "x-ratelimit-limit-requests-day"
LIMIT_TOKENS_MINUTE_HEADER = "x-ratelimit-limit-tokens-minute"
REMAINING_REQUESTS_DAY_HEADER = "x-ratelimit-remaining-requests-day"
REMAINING_TOKENS_MINUTE_HEADER = "x-ratelimit-remaining-tokens-minute"
RESET_REQUESTS_DAY_HEADER = "x-ratelimit-reset-requests-day"
RESET_TOKENS_MINUTE_HEADER = "x-ratelimit-reset-tokens-minute"


def _optional_int(value: str | None):
    if value is None:
        return None

    return int(value)


def _optional_float(value: str | None):
    if value is None:
        return None

    return float(value)


def _metadata(response: httpx.Response) -> ResponseMetadata:
    headers = response.headers
    return ResponseMetadata(
        status_code=response.status_code,
        request_id=headers.get(REQUEST_ID_HEADER),
        retry_after=_optional_float(headers.get(RETRY_AFTER_HEADER)),
        rate_limit=RateLimitState(
            limit_requests_day=_optional_int(headers.get(LIMIT_REQUESTS_DAY_HEADER)),
            limit_tokens_minute=_optional_int(headers.get(LIMIT_TOKENS_MINUTE_HEADER)),
            remaining_requests_day=_optional_int(headers.get(REMAINING_REQUESTS_DAY_HEADER)),
            remaining_tokens_minute=_optional_int(headers.get(REMAINING_TOKENS_MINUTE_HEADER)),
            reset_requests_day=_optional_float(headers.get(RESET_REQUESTS_DAY_HEADER)),
            reset_tokens_minute=_optional_float(headers.get(RESET_TOKENS_MINUTE_HEADER)),
        ),
    )


def _request_body(request: chat.CreateChatCompletionRequest):
    return request.model_dump(mode="json", by_alias=True, exclude_none=True)


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

    def __init__(self, *, api_key: str, http_client: httpx.Client) -> None:
        if not api_key:
            message = "api_key must not be empty"
            raise ValueError(message)
        self._http_client = http_client
        self._headers = {"authorization": f"Bearer {api_key}"}

    def create_chat_completion(
        self,
        request: chat.CreateChatCompletionRequest,
    ) -> APIResponse[chat.ChatCompletion]:
        """Execute one non-streaming chat completion request."""

        try:
            response = self._http_client.post(
                f"{BASE_URL}{chat.PATH}",
                json=_request_body(request),
                headers=self._headers,
            )
        except httpx.RequestError as exc:
            message = "Cerebras request failed before receiving a valid response"
            raise errors.TransportError(message) from exc

        metadata = _metadata(response)
        _raise_for_status(response, metadata)
        return APIResponse(data=_parse_response(response, metadata), metadata=metadata)
