from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EngineUsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 1
    prompt_breakdown: dict[str, int] = field(default_factory=dict)
    completion_breakdown: dict[str, int] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
