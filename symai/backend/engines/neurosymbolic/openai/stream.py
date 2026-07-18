"""OpenAI Responses SSE stream adapter: converts event-typed SSE into normalized deltas.

Unlike chat-completions streams, the Responses API emits named event types
(response.output_text.delta, response.reasoning_summary_text.delta, ...) and carries
the full final Response object in the terminal response.completed event.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from symai.backend.streaming import EngineStreamDelta

if TYPE_CHECKING:
    from symai.backend.transport import SSEEvent


class OpenAIStreamAdapter:
    def process_event(self, event: SSEEvent) -> EngineStreamDelta:
        if not event.data:
            return EngineStreamDelta(raw=event)
        if event.event in (
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ):
            return EngineStreamDelta(raw=event)

        chunk = json.loads(event.data)
        if event.event == "response.output_text.delta":
            return EngineStreamDelta(text=chunk.get("delta") or "", raw=chunk)
        if event.event == "response.reasoning_summary_text.delta":
            return EngineStreamDelta(thinking=chunk.get("delta") or "", raw=chunk)
        if event.event == "response.completed":
            response = chunk["response"]
            return EngineStreamDelta(usage=response.get("usage"), done=True, raw=response)
        if event.event in ("response.failed", "response.incomplete"):
            response = chunk.get("response", {})
            return EngineStreamDelta(
                usage=response.get("usage"),
                finish_reason=response.get("status"),
                done=True,
                raw=response,
            )

        return EngineStreamDelta(raw=chunk)
