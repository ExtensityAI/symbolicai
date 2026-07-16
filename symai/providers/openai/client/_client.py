import json
from typing import TypeVar

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

import symai.providers.openai.client.embeddings as embeddings
import symai.providers.openai.client.errors as errors
import symai.providers.openai.client.responses as responses_api
from symai.providers._client.headers import authorization_header
from symai.providers.openai.client.headers import extract_response_metadata
from symai.providers.openai.client.transport import APIResponse, ResponseMetadata

BASE_URL = "https://api.openai.com/v1"

T = TypeVar("T", bound=BaseModel)


def _raise_for_status(response: httpx.Response, metadata: ResponseMetadata):
    if response.status_code == httpx.codes.UNAUTHORIZED:
        raise errors.AuthError(metadata, response.text, "OpenAI API rejected credentials")
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        raise errors.RateLimitError(metadata, response.text, "OpenAI API rate limit exceeded")
    if not response.is_success:
        raise errors.APIError(metadata, response.text)


def _parse_response[T: BaseModel](
    response: httpx.Response, metadata: ResponseMetadata, model: type[T]
) -> T:
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
    """Synchronous owner of an OpenAI HTTP connection pool."""

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

    def _post(
        self,
        path: str,
        request: BaseModel,
        model: type[T],
    ) -> APIResponse[T]:
        json_body = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        try:
            response = self._http_client.post(
                f"{BASE_URL}{path}",
                json=json_body,
                headers=self._headers,
            )
        except httpx.RequestError as exc:
            message = "OpenAI request failed before receiving a valid response"
            raise errors.TransportError(message) from exc

        metadata = extract_response_metadata(response)
        _raise_for_status(response, metadata)
        return APIResponse(data=_parse_response(response, metadata, model), metadata=metadata)

    def create_response(
        self,
        request: responses_api.CreateResponseRequest,
    ) -> APIResponse[responses_api.Response]:
        return self._post(
            responses_api.PATH,
            request,
            responses_api.Response,
        )

    def create_embeddings(
        self,
        request: embeddings.CreateEmbeddingRequest,
    ) -> APIResponse[embeddings.EmbeddingList]:
        return self._post(embeddings.PATH, request, embeddings.EmbeddingList)
