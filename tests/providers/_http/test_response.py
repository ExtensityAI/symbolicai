import pytest
from pydantic import ValidationError

from symai.providers._http.response import APIResponse, HttpMetadata
from symai.providers.cerebras.client.response import (
    APIResponse as CerebrasAPIResponse,
)
from symai.providers.cerebras.client.response import (
    HttpMetadata as CerebrasHttpMetadata,
)
from symai.providers.cerebras.client.response import (
    RateLimitState,
)


def test_shared_response_envelope_is_strict_and_immutable():
    metadata = HttpMetadata(status_code=200, request_id="req-1", retry_after=None)
    response = APIResponse(data="ok", metadata=metadata)

    assert response.data == "ok"
    assert response.metadata is metadata
    with pytest.raises(ValidationError):
        HttpMetadata.model_validate(
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
    metadata = CerebrasHttpMetadata(
        status_code=200,
        request_id=None,
        retry_after=None,
        rate_limit=RateLimitState(),
    )
    response = CerebrasAPIResponse(data="ok", metadata=metadata)

    assert isinstance(metadata, HttpMetadata)
    assert CerebrasAPIResponse is APIResponse
    assert response.metadata is metadata
    assert metadata.rate_limit.remaining_tokens_minute is None
