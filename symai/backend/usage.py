from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens, locked at the provider models' API_PINNED date."""

    input: float
    output: float
    cached_input: float | None = None


@dataclass(frozen=True)
class EngineUsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 1
    prompt_breakdown: dict[str, int] = field(default_factory=dict)
    completion_breakdown: dict[str, int] = field(default_factory=dict)
    extras: dict[str, int] = field(default_factory=dict)
