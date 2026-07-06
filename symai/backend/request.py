from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict


class EngineRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


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
    params: dict[str, Any] | None = None
    timeout: float | None = None
    extra_body: dict[str, Any] | None = None

    def body(self) -> dict[str, Any]:
        body = {} if self.extra_body is None else dict(self.extra_body)
        payload_body = self.payload.model_dump(exclude_none=True)
        body.update(payload_body)
        return body

    def kwargs(self) -> dict[str, Any]:
        values = self.body()
        if self.call_options is not None:
            values.update(self.call_options.model_dump(exclude_none=True))
        return values
