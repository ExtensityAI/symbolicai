from __future__ import annotations

from dataclasses import dataclass
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

    def body(self) -> dict[str, Any]:
        return self.payload.model_dump(exclude_none=True)

    def kwargs(self) -> dict[str, Any]:
        values = self.body()
        if self.call_options is not None:
            values.update(self.call_options.model_dump(exclude_none=True))
        return values
