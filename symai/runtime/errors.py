from pydantic import Field

from symai.runtime.models import FrozenModel, NonNegativeFiniteFloat, Provider


class ErrorMetadata(FrozenModel):
    provider: Provider
    model: str = Field(min_length=1)
    request_id: str | None = None
    retry_after: NonNegativeFiniteFloat | None = None


class SymbolicAIRuntimeError(Exception):
    """Base exception for provider-neutral runtime failures."""


class NoActiveRuntimeError(SymbolicAIRuntimeError):
    """Raised when an operation requires an explicit active runtime."""


class RuntimeClosedError(SymbolicAIRuntimeError):
    """Raised when a runtime cannot accept work in its current lifecycle state."""


class UnsupportedCapabilityError(SymbolicAIRuntimeError):
    """Raised when a runtime does not own the requested capability."""


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
    """Raised when a provider rejects authentication."""


class RateLimitError(ExecutionError):
    """Raised when a provider rate-limits execution."""


class TransportError(ExecutionError):
    """Raised when provider transport fails before a valid response is available."""


class InvalidResponseError(ExecutionError):
    """Raised when a provider response cannot become a normalized response."""
