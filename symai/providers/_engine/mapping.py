from dataclasses import dataclass
from http import HTTPStatus
from typing import Never

from symai.providers._engine.base import retry_after_seconds
from symai.providers._http import errors as client_errors
from symai.runtime.errors import (
    AuthenticationError,
    ErrorMetadata,
    ExecutionError,
    InvalidRequestError,
    InvalidResponseError,
    PermissionDeniedError,
    ProviderError,
    RateLimitError,
    TransportError,
)
from symai.runtime.models import ProviderId

# A failure is worth retrying when it is about capacity or the provider's own health,
# never when the request itself is unacceptable. The library classifies but does not
# retry: automatic retries of a non-idempotent POST would risk duplicate billing.
_RETRYABLE_STATUS = frozenset(
    {
        HTTPStatus.REQUEST_TIMEOUT,
        HTTPStatus.TOO_MANY_REQUESTS,
        HTTPStatus.INTERNAL_SERVER_ERROR,
        HTTPStatus.BAD_GATEWAY,
        HTTPStatus.SERVICE_UNAVAILABLE,
        HTTPStatus.GATEWAY_TIMEOUT,
    }
)


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
        status_code = error.metadata.status_code
        message = messages.api.format(status_code=status_code)
        raise _api_error_type(status_code)(message, metadata=metadata) from error

    raise error


def _api_error_type(status_code: int) -> type[ExecutionError]:
    """Select the error class for a non-success status callers may react to differently."""
    if status_code == HTTPStatus.FORBIDDEN:
        return PermissionDeniedError
    if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return ProviderError
    if status_code >= HTTPStatus.BAD_REQUEST:
        return InvalidRequestError

    return ExecutionError


def _runtime_metadata(
    error: client_errors.ClientError,
    *,
    provider: ProviderId,
    model: str,
) -> ErrorMetadata:
    # Only an error raised from a response carries one to describe; a transport failure
    # never reached one. Narrowing states that, where `getattr(error, "metadata", None)`
    # only worked around the base class not declaring it and erased the type on the way.
    if isinstance(error, (client_errors.APIError, client_errors.ResponseError)):
        client_metadata = error.metadata
        details = error.details
    else:
        client_metadata = None
        details = None

    status_code = client_metadata.status_code if client_metadata is not None else None
    return ErrorMetadata(
        provider=provider,
        model=model,
        request_id=client_metadata.request_id if client_metadata is not None else None,
        retry_after=retry_after_seconds(
            client_metadata.retry_after if client_metadata is not None else None
        ),
        status_code=status_code,
        error_code=details.code if details is not None else None,
        error_type=details.type if details is not None else None,
        param=details.param if details is not None else None,
        provider_message=details.message if details is not None else None,
        retryable=status_code in _RETRYABLE_STATUS if status_code is not None else False,
    )
