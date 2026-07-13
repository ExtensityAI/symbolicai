import json
from typing import TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

from symai.clients._headers import parse_optional_float
import symai.clients.openai.embeddings as embeddings
import symai.clients.openai.errors as errors
import symai.clients.openai.responses as responses_api
from symai.clients.openai.transport import APIResponse, ResponseMetadata

BASE_URL = "https://api.openai.com/v1"
REQUEST_ID_HEADER = "x-request-id"
RETRY_AFTER_HEADER = "retry-after"

T = TypeVar("T", bound=BaseModel)


def _extract_response_metadata(response: httpx.Response) -> ResponseMetadata:
    return ResponseMetadata(
        status_code=response.status_code,
        request_id=response.headers.get(REQUEST_ID_HEADER),
        retry_after=parse_optional_float(response.headers.get(RETRY_AFTER_HEADER)),
    )


def _raise_for_status(response: httpx.Response, metadata: ResponseMetadata):
    if response.status_code == httpx.codes.UNAUTHORIZED:
        raise errors.AuthError(metadata, response.text, "OpenAI API rejected credentials")
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        raise errors.RateLimitError(metadata, response.text, "OpenAI API rate limit exceeded")
    if not response.is_success:
        raise errors.APIError(metadata, response.text)


def _parse_response(response: httpx.Response, metadata: ResponseMetadata, model: type[T]) -> T:
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

    def _request(
        self,
        method: str,
        path: str,
        model: type[T],
        *,
        body: BaseModel | None = None,
        query: BaseModel | None = None,
    ) -> APIResponse[T]:
        json_body = (
            body.model_dump(mode="json", by_alias=True, exclude_none=True)
            if body is not None
            else None
        )
        query_params = (
            query.model_dump(mode="json", exclude_none=True) if query is not None else None
        )

        try:
            http_response = self._http_client.request(
                method,
                f"{BASE_URL}{path}",
                headers=self._headers,
                json=json_body,
                params=query_params,
            )
        except httpx.RequestError as exc:
            message = "OpenAI request failed before receiving a valid response"
            raise errors.TransportError(message) from exc

        metadata = _extract_response_metadata(http_response)
        _raise_for_status(http_response, metadata)
        return APIResponse(
            data=_parse_response(http_response, metadata, model),
            metadata=metadata,
        )

    def _post(self, path: str, request: BaseModel, model: type[T]) -> APIResponse[T]:
        return self._request("POST", path, model, body=request)

    def create_response(
        self,
        request: responses_api.CreateResponseRequest,
    ) -> APIResponse[responses_api.Response]:
        return self._post(
            responses_api.PATH,
            request,
            responses_api.Response,
        )

    def retrieve_response(
        self,
        response_id: str,
        query: responses_api.RetrieveResponseParams | None = None,
    ) -> APIResponse[responses_api.Response]:
        response_path = f"{responses_api.PATH}/{quote(response_id, safe='')}"
        return self._request(
            "GET",
            response_path,
            responses_api.Response,
            query=query,
        )

    def delete_response(
        self,
        response_id: str,
    ) -> APIResponse[responses_api.DeletedResponse]:
        response_path = f"{responses_api.PATH}/{quote(response_id, safe='')}"
        return self._request(
            "DELETE",
            response_path,
            responses_api.DeletedResponse,
        )

    def cancel_response(
        self,
        response_id: str,
    ) -> APIResponse[responses_api.Response]:
        response_path = f"{responses_api.PATH}/{quote(response_id, safe='')}/cancel"
        return self._request(
            "POST",
            response_path,
            responses_api.Response,
        )

    def list_input_items(
        self,
        response_id: str,
        query: responses_api.ListInputItemsParams | None = None,
    ) -> APIResponse[responses_api.InputItemList]:
        response_path = f"{responses_api.PATH}/{quote(response_id, safe='')}/input_items"
        return self._request(
            "GET",
            response_path,
            responses_api.InputItemList,
            query=query,
        )

    def create_embeddings(
        self,
        request: embeddings.CreateEmbeddingRequest,
    ) -> APIResponse[embeddings.EmbeddingList]:
        return self._post(embeddings.PATH, request, embeddings.EmbeddingList)
