from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens, locked at the provider models' API_PINNED date."""

    input: float
    output: float
    cached_input: float | None = None
    # NOTE: the cache-WRITE rate where the provider bills writes above the plain
    # input rate (OpenAI GPT-5.6 and later: 1.25x input); None when a written
    # token costs the input rate.
    cache_write: float | None = None


@dataclass(frozen=True)
class EngineUsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 1
    prompt_breakdown: dict[str, int] = field(default_factory=dict)
    completion_breakdown: dict[str, int] = field(default_factory=dict)
    extras: dict[str, int] = field(default_factory=dict)
