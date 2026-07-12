from dataclasses import FrozenInstanceError

import pytest

from symai.backend.integrations.cerebras.chat import ChatResponse
from symai.backend.integrations.cerebras.response import (
    Metadata,
    RateLimitState,
    Response,
)


def test_response_carries_exact_data_and_metadata_objects():
    data = ChatResponse()
    metadata = Metadata(
        status_code=200,
        request_id="req-1",
        retry_after=2.5,
        rate_limit=RateLimitState(),
    )

    response = Response(data=data, metadata=metadata)

    assert response.data is data
    assert response.metadata is metadata


def test_response_values_are_immutable():
    metadata = Metadata(
        status_code=200,
        request_id=None,
        retry_after=None,
        rate_limit=RateLimitState(),
    )

    field = "status_code"
    with pytest.raises(FrozenInstanceError):
        setattr(metadata, field, 201)
