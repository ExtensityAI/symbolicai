import pytest
from pydantic import BaseModel, ValidationError

from symai.backend.integrations.cerebras.chat import ChatCompletion
from symai.backend.integrations.cerebras.transport import (
    APIResponse,
    RateLimitState,
    ResponseMetadata,
)


def test_response_carries_exact_data_and_metadata_objects():
    data = ChatCompletion()
    metadata = ResponseMetadata(
        status_code=200,
        request_id="req-1",
        retry_after=2.5,
        rate_limit=RateLimitState(),
    )

    response = APIResponse(data=data, metadata=metadata)

    assert response.data is data
    assert isinstance(response, BaseModel)
    assert isinstance(metadata, BaseModel)
    assert response.metadata is metadata


def test_response_values_are_immutable():
    metadata = ResponseMetadata(
        status_code=200,
        request_id=None,
        retry_after=None,
        rate_limit=RateLimitState(),
    )

    field = "status_code"
    with pytest.raises(ValidationError):
        setattr(metadata, field, 201)
