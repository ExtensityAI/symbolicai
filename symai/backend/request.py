from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, JsonValue


class EngineRequestPayload(BaseModel):
    # NOTE: populate_by_name lets aliased (camelCase wire) models validate snake_case
    # Python kwargs; fields without aliases are unaffected.
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class EngineResponsePayload(BaseModel):
    """Base for provider response models: freeze the snapshot and ignore fields the
    provider adds after API_PINNED, so responses never fail on forward-compatible input."""

    model_config = ConfigDict(extra="ignore", frozen=True)


Payload = TypeVar("Payload", bound=EngineRequestPayload)
CallOptions = TypeVar("CallOptions", bound=EngineRequestPayload)


@dataclass(frozen=True)
class EngineAPIRequest(Generic[Payload, CallOptions]):
    provider: str
    operation: str
    payload: Payload
    call_options: CallOptions | None = None
    method: str = "POST"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, JsonValue] | None = None
    timeout: float | None = None
    extra_body: dict[str, JsonValue] | None = None

    def body(self) -> dict[str, JsonValue]:
        body = {} if self.extra_body is None else dict(self.extra_body)
        # NOTE: by_alias lets camelCase APIs (Google) keep snake_case Python fields;
        # fields without aliases dump under their field name either way.
        payload_body = self.payload.model_dump(exclude_none=True, by_alias=True)
        body.update(payload_body)
        return body

    def kwargs(self) -> dict[str, JsonValue]:
        values = self.body()
        if self.call_options is not None:
            values.update(self.call_options.model_dump(exclude_none=True))
        return values
