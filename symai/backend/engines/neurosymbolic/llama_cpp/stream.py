"""llama.cpp SSE stream adapter: converts raw SSE events into normalized deltas."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from symai.backend.streaming import EngineStreamDelta

if TYPE_CHECKING:
    from symai.backend.transport import SSEEvent


class LlamaCppStreamAdapter:
    def process_event(self, event: SSEEvent) -> EngineStreamDelta:
        if event.data == "[DONE]":
            return EngineStreamDelta(done=True, raw=event)
        if not event.data:
            return EngineStreamDelta(raw=event)

        chunk = json.loads(event.data)
        usage = chunk["usage"] if chunk.get("usage") else None
        choices = chunk.get("choices") or []
        if not choices:
            return EngineStreamDelta(usage=usage, raw=chunk)

        choice = choices[0]
        delta = choice.get("delta") or {}
        return EngineStreamDelta(
            text=delta.get("content") or "",
            thinking=delta.get("reasoning_content") or "",
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            raw=chunk,
        )
