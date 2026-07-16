"""Typed HTTP response envelopes returned by provider clients."""

from symai.providers._client.models import StrictModel


class ResponseMetadata(StrictModel):
    status_code: int
    request_id: str | None
    retry_after: float | None


class APIResponse[DataT, MetadataT: ResponseMetadata](StrictModel):
    data: DataT
    metadata: MetadataT
