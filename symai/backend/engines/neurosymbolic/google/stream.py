"""Gemini SSE stream adapter: converts streamGenerateContent chunks into deltas.

Gemini streams GenerateContentResponse chunks as data-only SSE lines: text and
thought parts accumulate per chunk, and the final chunk carries usageMetadata.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from symai.backend.streaming import EngineStreamDelta

if TYPE_CHECKING:
    from symai.backend.transport import SSEEvent


class GoogleStreamAdapter:
    def process_event(self, event: SSEEvent) -> EngineStreamDelta:
        if event.data == "[DONE]":
            return EngineStreamDelta(done=True, raw=event)
        if not event.data:
            return EngineStreamDelta(raw=event)

        chunk = json.loads(event.data)
        usage = chunk.get("usageMetadata")
        candidates = chunk.get("candidates") or []
        if not candidates:
            # NOTE: the terminal chunk usually carries usageMetadata without candidates.
            return EngineStreamDelta(usage=usage, done=usage is not None, raw=chunk)

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = ""
        thinking = ""
        for part in parts:
            if part.get("thought"):
                thinking += part.get("text") or ""
            else:
                text += part.get("text") or ""

        return EngineStreamDelta(
            text=text,
            thinking=thinking,
            usage=usage,
            finish_reason=candidate.get("finishReason"),
            done=usage is not None,
            raw=chunk,
        )
