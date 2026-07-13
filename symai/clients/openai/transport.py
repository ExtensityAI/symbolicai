from typing import Generic, TypeVar

from symai.clients._models import StrictModel

T = TypeVar("T")


class ResponseMetadata(StrictModel):
    status_code: int
    request_id: str | None
    retry_after: float | None


class APIResponse(StrictModel, Generic[T]):
    data: T
    metadata: ResponseMetadata
