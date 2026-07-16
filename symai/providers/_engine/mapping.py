from dataclasses import dataclass
from typing import Never, Protocol, cast

from symai.providers._client import errors as client_errors
from symai.providers._engine.base import retry_after_seconds
from symai.runtime.errors import (
    AuthenticationError,
    ErrorMetadata,
    ExecutionError,
    InvalidResponseError,
    RateLimitError,
    TransportError,
)
from symai.runtime.models import ProviderId


class _ResponseMetadata(Protocol):
    status_code: int
    request_id: str | None
    retry_after: float | None


@dataclass(frozen=True, slots=True)
class ClientErrorMessages:
    authentication: str
    rate_limit: str
    response: str
    transport: str
    api: str


def raise_mapped_client_error(
    error: client_errors.ClientError,
    *,
    provider: ProviderId,
    model: str,
    messages: ClientErrorMessages,
) -> Never:
    metadata = _runtime_metadata(error, provider=provider, model=model)

    if isinstance(error, client_errors.AuthError):
        raise AuthenticationError(messages.authentication, metadata=metadata) from error
    if isinstance(error, client_errors.RateLimitError):
        raise RateLimitError(messages.rate_limit, metadata=metadata) from error
    if isinstance(error, client_errors.ResponseError):
        raise InvalidResponseError(messages.response, metadata=metadata) from error
    if isinstance(error, client_errors.TransportError):
        raise TransportError(messages.transport, metadata=metadata) from error
    if isinstance(error, client_errors.APIError):
        client_metadata = cast("_ResponseMetadata", error.metadata)
        message = messages.api.format(status_code=client_metadata.status_code)
        raise ExecutionError(message, metadata=metadata) from error

    raise error


def _runtime_metadata(
    error: client_errors.ClientError,
    *,
    provider: ProviderId,
    model: str,
) -> ErrorMetadata:
    client_metadata = cast("_ResponseMetadata | None", getattr(error, "metadata", None))
    return ErrorMetadata(
        provider=provider,
        model=model,
        request_id=client_metadata.request_id if client_metadata is not None else None,
        retry_after=retry_after_seconds(
            client_metadata.retry_after if client_metadata is not None else None
        ),
    )
