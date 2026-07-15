import pytest
from pydantic import ValidationError

import symai.runtime.errors as errors_module
from symai.runtime.errors import (
    AuthenticationError,
    ErrorMetadata,
    ExecutionError,
    InvalidResponseError,
    RateLimitError,
    RuntimeClosedError,
    SymbolicAIRuntimeError,
    TransportError,
    UnsupportedCapabilityError,
    UnsupportedFeatureError,
    UnsupportedModelError,
)

ENGINE_ERROR_NAMES = (
    "UnknownEngineError",
    "EngineCapabilityError",
    "AmbiguousEngineError",
    "RuntimeOwnershipError",
)


@pytest.mark.parametrize(
    "error_type",
    [
        RuntimeClosedError,
        UnsupportedCapabilityError,
        UnsupportedModelError,
        UnsupportedFeatureError,
        ExecutionError,
        AuthenticationError,
        RateLimitError,
        TransportError,
        InvalidResponseError,
    ],
)
def test_all_runtime_errors_share_provider_neutral_base(error_type):
    assert issubclass(error_type, SymbolicAIRuntimeError)


def test_ambient_runtime_error_is_absent() -> None:
    assert not hasattr(errors_module, "NoActiveRuntimeError")


@pytest.mark.parametrize("error_name", ENGINE_ERROR_NAMES)
def test_runtime_selection_and_ownership_errors_share_runtime_base(
    error_name: str,
) -> None:
    error_type = getattr(errors_module, error_name, None)

    assert error_type is not None
    assert issubclass(error_type, SymbolicAIRuntimeError)


def test_unknown_engine_error_has_safe_structured_payload() -> None:
    error = errors_module.UnknownEngineError("missing")

    assert error.engine_name == "missing"
    assert vars(error) == {"engine_name": "missing"}
    assert "missing" in str(error)


def test_engine_capability_error_has_safe_structured_payload() -> None:
    error = errors_module.EngineCapabilityError(
        "vectors",
        requested_capability="language_model",
        engine_capability="embedding",
    )

    assert error.engine_name == "vectors"
    assert error.requested_capability == "language_model"
    assert error.engine_capability == "embedding"
    assert vars(error) == {
        "engine_name": "vectors",
        "requested_capability": "language_model",
        "engine_capability": "embedding",
    }


def test_ambiguous_engine_error_freezes_sorted_safe_names() -> None:
    names = ["zeta", "alpha"]
    error = errors_module.AmbiguousEngineError("language_model", names)
    names.append("secret-that-was-not-part-of-the-error")

    assert error.capability == "language_model"
    assert error.engine_names == ("alpha", "zeta")
    assert vars(error) == {
        "capability": "language_model",
        "engine_names": ("alpha", "zeta"),
    }
    assert str(error).endswith("alpha, zeta")


def test_runtime_ownership_error_does_not_expose_thread_identifiers() -> None:
    error = errors_module.RuntimeOwnershipError("execute")

    assert error.operation == "execute"
    assert vars(error) == {"operation": "execute"}
    assert str(error) == "Runtime execute must run on its owner thread"


@pytest.mark.parametrize(
    "error_type",
    [AuthenticationError, RateLimitError, TransportError, InvalidResponseError],
)
def test_provider_execution_errors_share_execution_base(error_type):
    assert issubclass(error_type, ExecutionError)


def test_execution_error_carries_frozen_normalized_metadata():
    metadata = ErrorMetadata(
        provider="cerebras",
        model="gpt-oss-120b",
        request_id="req-1",
        retry_after=2.5,
    )
    error = RateLimitError("rate limit exceeded", metadata=metadata)

    assert str(error) == "rate limit exceeded"
    assert error.metadata is metadata
    assert not hasattr(error, "raw_response")
    assert not hasattr(error, "response_body")
    with pytest.raises(ValidationError):
        metadata.request_id = "changed"


def test_execution_error_metadata_is_optional():
    error = TransportError("connection failed")

    assert error.metadata is None


def test_error_metadata_has_open_provider_ids_and_finite_non_negative_retry_after():
    metadata = ErrorMetadata.model_validate({"provider": "ACME_Local", "model": "gpt-5.5"})
    assert metadata.provider == "acme_local"
    with pytest.raises(ValidationError):
        ErrorMetadata(provider="invalid provider", model="gpt-5.5")
    with pytest.raises(ValidationError):
        ErrorMetadata(provider="openai", model="gpt-5.5", retry_after=-0.1)
    with pytest.raises(ValidationError):
        ErrorMetadata(provider="openai", model="gpt-5.5", retry_after=float("inf"))
    with pytest.raises(ValidationError):
        ErrorMetadata.model_validate(
            {"provider": "openai", "model": "gpt-5.5", "unknown": "raw"}
        )
