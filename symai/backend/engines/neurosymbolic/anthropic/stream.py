"""Anthropic SSE stream adapter: converts event-typed SSE into normalized deltas.

Anthropic streams named events (message_start, content_block_start/delta/stop,
message_delta, message_stop, ping). Usage arrives split: input tokens on
message_start, output tokens on message_delta — the engine merges both.

Each content block is accumulated in emit order, including thinking signatures
(signature_delta) and tool_use JSON (input_json_delta). The collector must be
able to send those blocks back on the next turn.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from symai.backend.streaming import EngineStreamDelta

if TYPE_CHECKING:
    from symai.backend.transport import SSEEvent


class AnthropicStreamAdapter:
    def __init__(self):
        self.blocks = {}
        self.tool_calls = []
        self._tool_json = {}

    def content(self) -> list[dict]:
        """Completed content blocks in Anthropic index order."""
        return [self.blocks[index] for index in sorted(self.blocks)]

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
            return self._start_block(chunk)
        if event.event == "content_block_delta":
            return self._delta_block(chunk)
        if event.event == "content_block_stop":
            return self._stop_block(chunk)
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

    def _start_block(self, chunk: dict) -> EngineStreamDelta:
        index = chunk["index"]
        block = dict(chunk.get("content_block") or {})
        self.blocks[index] = block
        if block.get("type") in {"tool_use", "server_tool_use"}:
            self._tool_json[index] = ""
        if block.get("type") == "text":
            return EngineStreamDelta(text=block.get("text") or "", raw=chunk)
        if block.get("type") == "thinking":
            return EngineStreamDelta(thinking=block.get("thinking") or "", raw=chunk)
        return EngineStreamDelta(raw=chunk)

    def _delta_block(self, chunk: dict) -> EngineStreamDelta:
        index = chunk.get("index")
        delta = chunk.get("delta") or {}
        delta_type = delta.get("type")
        block = self.blocks.get(index)
        if block is None:
            return EngineStreamDelta(raw=chunk)
        if delta_type == "text_delta":
            text = delta.get("text") or ""
            block["text"] = (block.get("text") or "") + text
            return EngineStreamDelta(text=text, raw=chunk)
        if delta_type == "thinking_delta":
            thinking = delta.get("thinking") or ""
            block["thinking"] = (block.get("thinking") or "") + thinking
            return EngineStreamDelta(thinking=thinking, raw=chunk)
        if delta_type == "signature_delta":
            # NOTE: the stamp Anthropic requires when the thinking block is echoed
            # back on the next turn. One or more deltas concatenate to the signature.
            block["signature"] = (block.get("signature") or "") + (delta.get("signature") or "")
            return EngineStreamDelta(raw=chunk)
        if delta_type == "input_json_delta" and index in self._tool_json:
            self._tool_json[index] += delta.get("partial_json") or ""
        return EngineStreamDelta(raw=chunk)

    def _stop_block(self, chunk: dict) -> EngineStreamDelta:
        index = chunk.get("index")
        if index in self._tool_json:
            raw_json = self._tool_json.pop(index)
            block = self.blocks[index]
            if raw_json:
                try:
                    block["input"] = json.loads(raw_json)
                except json.JSONDecodeError:
                    block["input"] = {}
            elif "input" not in block:
                block["input"] = {}
            if block.get("type") == "tool_use":
                # NOTE: server_tool_use streams input the same way, but the server
                # already executed it — keep it out of the client-side call surface.
                self.tool_calls.append(
                    {"id": block.get("id"), "name": block.get("name"), "input": block.get("input")}
                )
        return EngineStreamDelta(raw=chunk)
