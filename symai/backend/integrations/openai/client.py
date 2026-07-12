import json
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from symai.backend.integrations.openai import embeddings, errors, responses
from symai.backend.integrations.openai.response import Metadata, Response

BASE_URL = "https://api.openai.com/v1"
REQUEST_ID_HEADER = "x-request-id"
RETRY_AFTER_HEADER = "retry-after"

T = TypeVar("T", bound=BaseModel)


def _optional_float(value: str | None):
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _metadata(response: httpx.Response) -> Metadata:
    return Metadata(
        status_code=response.status_code,
        request_id=response.headers.get(REQUEST_ID_HEADER),
        retry_after=_optional_float(response.headers.get(RETRY_AFTER_HEADER)),
    )


def _raise_for_status(response: httpx.Response, metadata: Metadata):
    if response.status_code == httpx.codes.UNAUTHORIZED:
        raise errors.AuthError(metadata, response.text, "OpenAI API rejected credentials")
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        raise errors.RateLimitError(metadata, response.text, "OpenAI API rate limit exceeded")
    if not response.is_success:
        raise errors.APIError(metadata, response.text)


def _parse_response(response: httpx.Response, metadata: Metadata, model: type[T]) -> T:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        message = "OpenAI response was not valid JSON"
        raise errors.ResponseError(message, metadata=metadata, body=response.text) from exc

    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        message = "OpenAI response did not match the expected schema"
        raise errors.ResponseError(message, metadata=metadata, body=response.text) from exc


class Client:
    """Synchronous caller-owned client for OpenAI Responses and Embeddings."""

    def __init__(self, *, api_key: str, http_client: httpx.Client) -> None:
        if not api_key:
            message = "api_key must not be empty"
            raise ValueError(message)
        self._http_client = http_client
        self._headers = {"authorization": f"Bearer {api_key}"}

    def _post(self, path: str, request: BaseModel, model: type[T]) -> Response[T]:
        try:
            http_response = self._http_client.post(
                f"{BASE_URL}{path}",
                json=request.model_dump(mode="json", exclude_none=True),
                headers=self._headers,
            )
        except httpx.RequestError as exc:
            message = "OpenAI request failed before receiving a valid response"
            raise errors.TransportError(message) from exc

        metadata = _metadata(http_response)
        _raise_for_status(http_response, metadata)
        return Response(
            data=_parse_response(http_response, metadata, model),
            metadata=metadata,
        )

    def responses(
        self,
        request: responses.ResponsesRequest,
    ) -> Response[responses.ResponsesResponse]:
        return self._post(responses.PATH, request, responses.ResponsesResponse)

    def embeddings(
        self,
        request: embeddings.EmbeddingRequest,
    ) -> Response[embeddings.EmbeddingResponse]:
        return self._post(embeddings.PATH, request, embeddings.EmbeddingResponse)
