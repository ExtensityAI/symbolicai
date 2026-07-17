from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import JsonValue


@dataclass(frozen=True)
class EngineStreamDelta:
    text: str = ""
    thinking: str = ""
    usage: dict[str, JsonValue] | None = None
    finish_reason: str | None = None
    done: bool = False
    raw: object = None


@dataclass
class EngineStreamAccumulator:
    text_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    usage: dict[str, JsonValue] | None = None
    finish_reason: str | None = None
    done: bool = False

    def add(self, delta: EngineStreamDelta) -> None:
        if delta.text:
            self.text_parts.append(delta.text)
        if delta.thinking:
            self.thinking_parts.append(delta.thinking)
        if delta.usage is not None:
            self.usage = delta.usage
        if delta.finish_reason is not None:
            self.finish_reason = delta.finish_reason
        if delta.done:
            self.done = True

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def thinking(self) -> str:
        return "".join(self.thinking_parts)
