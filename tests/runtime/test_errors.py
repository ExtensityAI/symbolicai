import pytest
from pydantic import ValidationError

from symai.runtime.errors import (
    AuthenticationError,
    ErrorMetadata,
    ExecutionError,
    InvalidResponseError,
    NoActiveRuntimeError,
    RateLimitError,
    RuntimeClosedError,
    SymbolicAIRuntimeError,
    TransportError,
    UnsupportedCapabilityError,
    UnsupportedFeatureError,
    UnsupportedModelError,
)
from symai.runtime.models import Provider


@pytest.mark.parametrize(
    "error_type",
    [
        NoActiveRuntimeError,
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


@pytest.mark.parametrize(
    "error_type",
    [AuthenticationError, RateLimitError, TransportError, InvalidResponseError],
)
def test_provider_execution_errors_share_execution_base(error_type):
    assert issubclass(error_type, ExecutionError)


def test_execution_error_carries_frozen_normalized_metadata():
    metadata = ErrorMetadata(
        provider=Provider.CEREBRAS,
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


def test_error_metadata_is_strict_and_retry_after_is_finite_non_negative():
    with pytest.raises(ValidationError):
        ErrorMetadata.model_validate({"provider": "openai", "model": "gpt-5.5"})
    with pytest.raises(ValidationError):
        ErrorMetadata(provider=Provider.OPENAI, model="gpt-5.5", retry_after=-0.1)
    with pytest.raises(ValidationError):
        ErrorMetadata(provider=Provider.OPENAI, model="gpt-5.5", retry_after=float("inf"))
    with pytest.raises(ValidationError):
        ErrorMetadata.model_validate(
            {"provider": Provider.OPENAI, "model": "gpt-5.5", "unknown": "raw"}
        )
