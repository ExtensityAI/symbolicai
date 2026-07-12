from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Metadata:
    status_code: int
    request_id: str | None
    retry_after: float | None


@dataclass(frozen=True, slots=True)
class Response(Generic[T]):
    data: T
    metadata: Metadata
