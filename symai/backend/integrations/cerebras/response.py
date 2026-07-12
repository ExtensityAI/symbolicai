from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RateLimitState:
    limit_requests_day: int | None = None
    limit_tokens_minute: int | None = None
    remaining_requests_day: int | None = None
    remaining_tokens_minute: int | None = None
    reset_requests_day: float | None = None
    reset_tokens_minute: float | None = None


@dataclass(frozen=True, slots=True)
class Metadata:
    status_code: int
    request_id: str | None
    retry_after: float | None
    rate_limit: RateLimitState


@dataclass(frozen=True, slots=True)
class Response(Generic[T]):
    data: T
    metadata: Metadata
