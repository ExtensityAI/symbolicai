"""Anthropic SSE stream adapter: converts event-typed SSE into normalized deltas.

Anthropic streams named events (message_start, content_block_start/delta/stop,
message_delta, message_stop, ping). Usage arrives split: input tokens on
message_start, output tokens on message_delta — the engine merges both. Tool use
arrives as tool_use blocks whose JSON input streams via input_json_delta events;
the adapter is stateful and accumulates them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from symai.backend.streaming import EngineStreamDelta

if TYPE_CHECKING:
    from symai.backend.transport import SSEEvent


class AnthropicStreamAdapter:
    def __init__(self):
        self.tool_calls = []
        self._active_tool_calls = {}

    def process_event(self, event: SSEEvent) -> EngineStreamDelta:
        if event.data == "[DONE]":
            # NOTE: Anthropic streams terminate with message_stop, but tolerate the
            # OpenAI-style sentinel so a stray [DONE] never crashes the collector.
            return EngineStreamDelta(done=True, raw=event)
        if not event.data:
            return EngineStreamDelta(raw=event)
        if event.event == "ping":
            return EngineStreamDelta(raw=event)

        chunk = json.loads(event.data)
        if event.event == "message_start":
            usage = chunk.get("message", {}).get("usage")
            return EngineStreamDelta(usage=usage, raw=chunk)
        if event.event == "content_block_start":
            block = chunk.get("content_block", {})
            if block.get("type") == "tool_use":
                self._active_tool_calls[chunk["index"]] = {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "json": "",
                }
            return EngineStreamDelta(raw=chunk)
        if event.event == "content_block_delta":
            delta = chunk.get("delta", {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                return EngineStreamDelta(text=delta.get("text") or "", raw=chunk)
            if delta_type == "thinking_delta":
                return EngineStreamDelta(thinking=delta.get("thinking") or "", raw=chunk)
            if delta_type == "input_json_delta" and chunk.get("index") in self._active_tool_calls:
                self._active_tool_calls[chunk["index"]]["json"] += delta.get("partial_json") or ""
            return EngineStreamDelta(raw=chunk)
        if event.event == "content_block_stop":
            index = chunk.get("index")
            if index in self._active_tool_calls:
                info = self._active_tool_calls.pop(index)
                try:
                    arguments = json.loads(info["json"]) if info["json"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                self.tool_calls.append({"id": info["id"], "name": info["name"], "input": arguments})
            return EngineStreamDelta(raw=chunk)
        if event.event == "message_delta":
            delta = chunk.get("delta") or {}
            return EngineStreamDelta(
                usage=chunk.get("usage"),
                finish_reason=delta.get("stop_reason"),
                raw=chunk,
            )
        if event.event == "message_stop":
            return EngineStreamDelta(done=True, raw=chunk)

        return EngineStreamDelta(raw=chunk)
