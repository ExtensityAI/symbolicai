"""Typed HTTP response envelopes returned by the OpenAI client."""

from typing import TypeVar

from symai.providers._client.models import StrictModel

T = TypeVar("T")


class ResponseMetadata(StrictModel):
    status_code: int
    request_id: str | None
    retry_after: float | None


class APIResponse[T](StrictModel):
    data: T
    metadata: ResponseMetadata
