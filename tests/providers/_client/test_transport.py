import pytest
from pydantic import ValidationError

from symai.providers._client.transport import APIResponse, ResponseMetadata
from symai.providers.cerebras.client.transport import (
    APIResponse as CerebrasAPIResponse,
)
from symai.providers.cerebras.client.transport import (
    RateLimitState,
)
from symai.providers.cerebras.client.transport import (
    ResponseMetadata as CerebrasResponseMetadata,
)


def test_shared_response_envelope_is_strict_and_immutable():
    metadata = ResponseMetadata(status_code=200, request_id="req-1", retry_after=None)
    response = APIResponse(data="ok", metadata=metadata)

    assert response.data == "ok"
    assert response.metadata is metadata
    with pytest.raises(ValidationError):
        ResponseMetadata.model_validate(
            {
                "status_code": 200,
                "request_id": None,
                "retry_after": None,
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError):
        response.data = "changed"


def test_cerebras_metadata_adds_rate_limit_state_to_shared_metadata():
    metadata = CerebrasResponseMetadata(
        status_code=200,
        request_id=None,
        retry_after=None,
        rate_limit=RateLimitState(),
    )
    response = CerebrasAPIResponse(data="ok", metadata=metadata)

    assert isinstance(metadata, ResponseMetadata)
    assert CerebrasAPIResponse is APIResponse
    assert response.metadata is metadata
    assert metadata.rate_limit.remaining_tokens_minute is None
