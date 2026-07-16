"""Typed HTTP response envelopes returned by provider clients."""

from symai.providers._http.schema import StrictModel


class HttpMetadata(StrictModel):
    status_code: int
    request_id: str | None
    retry_after: float | None


class APIResponse[DataT, MetadataT: HttpMetadata](StrictModel):
    data: DataT
    metadata: MetadataT
