"""Typed HTTP response envelopes and rate-limit metadata returned by Cerebras."""

from typing import Generic, TypeVar

from symai.providers._client.models import StrictModel

T = TypeVar("T")


class RateLimitState(StrictModel):
    limit_requests_day: int | None = None
    limit_tokens_minute: int | None = None
    remaining_requests_day: int | None = None
    remaining_tokens_minute: int | None = None
    reset_requests_day: float | None = None
    reset_tokens_minute: float | None = None


class ResponseMetadata(StrictModel):
    status_code: int
    request_id: str | None
    retry_after: float | None
    rate_limit: RateLimitState


class APIResponse(StrictModel, Generic[T]):
    data: T
    metadata: ResponseMetadata
