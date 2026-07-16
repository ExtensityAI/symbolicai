from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from symai.runtime.models import FrozenModel, NonNegativeFiniteFloat, ProviderId

type EngineCapability = Literal["language_model", "embedding"]


class ErrorMetadata(FrozenModel):
    """Safe, bounded facts about a failed provider call.

    Every field is either application-owned or a short provider identifier. Prompts,
    credentials, and raw provider payloads never appear here; `retryable` reports whether
    the failure class is worth retrying, but the library never retries on its own.
    """

    provider: ProviderId
    model: str = Field(min_length=1)
    request_id: str | None = None
    retry_after: NonNegativeFiniteFloat | None = None
    status_code: int | None = None
    error_code: str | None = None
    error_type: str | None = None
    param: str | None = None
    provider_message: str | None = None
    retryable: bool = False


class SymbolicAIRuntimeError(Exception):
    """Base exception for provider-neutral runtime failures."""


class RuntimeClosedError(SymbolicAIRuntimeError):
    """Raised when a runtime cannot accept work in its current lifecycle state."""


class UnsupportedCapabilityError(SymbolicAIRuntimeError):
    """Raised when a runtime does not own the requested capability."""


class UnknownEngineError(SymbolicAIRuntimeError):
    """Raised when an explicit engine name is not configured."""

    def __init__(self, engine_name: str) -> None:
        self.engine_name = engine_name
        msg = f"Runtime has no engine named {engine_name!r}"
        super().__init__(msg)


class EngineCapabilityError(SymbolicAIRuntimeError):
    """Raised when a named engine cannot satisfy the requested capability."""

    def __init__(
        self,
        engine_name: str,
        *,
        requested_capability: EngineCapability,
        engine_capability: EngineCapability,
    ) -> None:
        self.engine_name = engine_name
        self.requested_capability = requested_capability
        self.engine_capability = engine_capability
        msg = f"Engine {engine_name!r} provides {engine_capability}, not {requested_capability}"
        super().__init__(msg)


class AmbiguousEngineError(SymbolicAIRuntimeError):
    """Raised when implicit selection has multiple matching engines."""

    def __init__(
        self,
        capability: EngineCapability,
        engine_names: Sequence[str],
    ) -> None:
        self.capability = capability
        self.engine_names = tuple(sorted(engine_names))
        names = ", ".join(self.engine_names)
        msg = f"Runtime has multiple {capability} engines; select one of: {names}"
        super().__init__(msg)


class RuntimeOwnershipError(SymbolicAIRuntimeError):
    """Raised when an owner-bound Runtime is used from another thread."""

    def __init__(self, operation: Literal["enter", "execute", "close", "exit"]) -> None:
        self.operation = operation
        msg = f"Runtime {operation} must run on its owner thread"
        super().__init__(msg)


class UnsupportedModelError(SymbolicAIRuntimeError):
    """Raised when a model is absent from the selected provider catalog."""


class UnsupportedFeatureError(SymbolicAIRuntimeError):
    """Raised before transport when a model cannot satisfy a normalized request."""


class ExecutionError(SymbolicAIRuntimeError):
    """Base exception for failures while executing a provider request."""

    def __init__(self, message: str, *, metadata: ErrorMetadata | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata


class AuthenticationError(ExecutionError):
    """Raised when a provider rejects the credentials themselves."""


class PermissionDeniedError(ExecutionError):
    """Raised when credentials are valid but not entitled to the request."""


class InvalidRequestError(ExecutionError):
    """Raised when a provider rejects the request as malformed or unacceptable."""


class ProviderError(ExecutionError):
    """Raised when a provider fails on its own side."""


class RateLimitError(ExecutionError):
    """Raised when a provider rate-limits execution."""


class TransportError(ExecutionError):
    """Raised when provider transport fails before a valid response is available."""


class InvalidResponseError(ExecutionError):
    """Raised when a provider response cannot become a normalized response."""
