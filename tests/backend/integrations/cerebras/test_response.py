from dataclasses import FrozenInstanceError

import pytest

from symai.backend.integrations.cerebras.chat import ChatResponse
from symai.backend.integrations.cerebras.response import (
    RateLimitState,
    Response,
    ResponseMetadata,
)


def test_response_carries_body_and_http_metadata():
    metadata = ResponseMetadata(
        status_code=200,
        request_id="req-1",
        retry_after=2.5,
        rate_limit=RateLimitState(
            limit_requests_day=100,
            limit_tokens_minute=1_000,
            remaining_requests_day=99,
            remaining_tokens_minute=900,
            reset_requests_day=30.5,
            reset_tokens_minute=5.5,
        ),
    )
    response = Response(data=ChatResponse(), metadata=metadata)

    assert response.data == ChatResponse()
    assert response.metadata.request_id == "req-1"
    assert response.metadata.rate_limit.remaining_tokens_minute == 900


def test_response_values_are_immutable():
    metadata = ResponseMetadata(
        status_code=200,
        request_id=None,
        retry_after=None,
        rate_limit=RateLimitState(),
    )

    with pytest.raises(FrozenInstanceError):
        setattr(metadata, "status_code", 201)
