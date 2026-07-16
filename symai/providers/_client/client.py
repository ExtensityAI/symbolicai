import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from symai.providers._client import errors
from symai.providers._client.headers import authorization_header
from symai.providers._client.settings import HttpProviderSettings
from symai.providers._client.transport import APIResponse, ResponseMetadata


@dataclass(frozen=True, slots=True)
class ClientConfig[MetadataT: ResponseMetadata]:
    base_url: str
    provider_name: str
    extract_response_metadata: Callable[[httpx.Response], MetadataT]
    api_error: type[errors.APIError]
    auth_error: type[errors.AuthError]
    rate_limit_error: type[errors.RateLimitError]
    response_error: type[errors.ResponseError]
    transport_error: type[errors.TransportError]


class BaseClient[MetadataT: ResponseMetadata]:
    """Synchronous owner of an HTTP provider connection pool.

    `MetadataT` is the provider's response-metadata type. Parameterizing it keeps a
    provider that extends `ResponseMetadata` (extra rate-limit fields, say) visible to
    callers of `_post` instead of widening it back to the shared base.
    """

    config: ClientConfig[MetadataT]

    def __init__(
        self,
        *,
        api_key: SecretStr,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout | float = 5.0,
        connect_retries: int = 0,
    ) -> None:
        authorization = authorization_header(api_key)
        owned_transport = None
        if transport is None:
            owned_transport = httpx.HTTPTransport(retries=connect_retries)
        elif connect_retries:
            msg = "connect_retries cannot be combined with an injected transport"
            raise ValueError(msg)

        try:
            http_client = httpx.Client(timeout=timeout, transport=transport or owned_transport)
        except BaseException as error:
            # Only a transport this client created may be closed here; an injected one
            # belongs to the caller and may outlive a failed construction.
            if owned_transport is not None:
                try:
                    owned_transport.close()
                except BaseException as cleanup_error:
                    error.add_note(f"Client construction cleanup failed: {cleanup_error!r}")
            raise

        self._http_client = http_client
        self._headers = {"authorization": authorization}
        self._closed = False

    @classmethod
    def from_settings(cls, settings: HttpProviderSettings) -> Self:
        """Build a client from the settings type it defines.

        Composing a timeout out of a request and connect budget is HTTP detail, so it
        lives with the transport that understands it. An engine abstracts over its client
        and must not know a transport exists at all — an integration that speaks to a
        local binary rather than an API has no timeout to compose.
        """
        return cls(
            api_key=settings.api_key,
            timeout=httpx.Timeout(
                settings.request_timeout,
                connect=settings.connect_timeout,
            ),
            connect_retries=settings.connect_retries,
        )

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._http_client.close()

    def _post[T: BaseModel](
        self,
        path: str,
        request: BaseModel,
        model: type[T],
    ) -> APIResponse[T, MetadataT]:
        config = self.config
        json_body = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        try:
            response = self._http_client.post(
                f"{config.base_url}{path}",
                json=json_body,
                headers=self._headers,
            )
        except httpx.RequestError as exc:
            message = f"{config.provider_name} request failed before receiving a valid response"
            raise config.transport_error(message) from exc

        metadata = config.extract_response_metadata(response)
        self._raise_for_status(response, metadata)
        data = self._parse_response(response, metadata, model)
        return APIResponse(data=data, metadata=metadata)

    def _raise_for_status(
        self,
        response: httpx.Response,
        metadata: ResponseMetadata,
    ) -> None:
        config = self.config
        if response.status_code == httpx.codes.UNAUTHORIZED:
            message = f"{config.provider_name} API rejected credentials"
            raise config.auth_error(metadata, response.text, message)
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            message = f"{config.provider_name} API rate limit exceeded"
            raise config.rate_limit_error(metadata, response.text, message)
        if not response.is_success:
            raise config.api_error(metadata, response.text)

    def _parse_response[T: BaseModel](
        self,
        response: httpx.Response,
        metadata: ResponseMetadata,
        model: type[T],
    ) -> T:
        config = self.config
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            message = f"{config.provider_name} response was not valid JSON"
            raise config.response_error(
                message,
                metadata=metadata,
                body=response.text,
            ) from exc

        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            message = f"{config.provider_name} response did not match the expected schema"
            raise config.response_error(
                message,
                metadata=metadata,
                body=response.text,
            ) from exc
