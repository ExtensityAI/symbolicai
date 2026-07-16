"""Typed HTTP response envelopes and rate-limit metadata returned by Cerebras."""

from symai.providers._client import transport as _transport
from symai.providers._client.models import StrictModel
from symai.providers._client.transport import ResponseMetadata as BaseResponseMetadata


class RateLimitState(StrictModel):
    limit_requests_day: int | None = None
    limit_tokens_minute: int | None = None
    remaining_requests_day: int | None = None
    remaining_tokens_minute: int | None = None
    reset_requests_day: float | None = None
    reset_tokens_minute: float | None = None


class ResponseMetadata(BaseResponseMetadata):
    rate_limit: RateLimitState


APIResponse = _transport.APIResponse
