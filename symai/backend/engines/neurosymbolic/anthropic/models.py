"""Anthropic Messages API wire models.

Locked against https://platform.claude.com/docs/en/api/messages
Pricing: https://platform.claude.com/docs/en/about-claude/pricing
Version header: https://platform.claude.com/docs/en/api/versioning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload
from symai.backend.usage import ModelPricing
from symai.prompts import split_cache_breakpoints

API_PINNED = "2026-07-17"
ANTHROPIC_VERSION = "2023-06-01"

CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}
LONG_CONTEXT_1M_TOKENS = 1_000_000
LONG_CONTEXT_1M_BETA_HEADER = "context-1m-2025-08-07"
MAX_CACHE_BREAKPOINTS = 4


@dataclass(frozen=True)
class AnthropicModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    adaptive_thinking: bool
    long_context_1m: bool
    default_long_context_1m: bool
    sampling: bool
    pricing: ModelPricing | None


ANTHROPIC_MODEL_SPECS = {
    "claude-fable-5": AnthropicModelSpec(
        context_tokens=LONG_CONTEXT_1M_TOKENS,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        # NOTE: adaptive thinking is always on for Fable; temperature/top_p/top_k are
        # rejected as deprecated (verified 400 at API_PINNED).
        adaptive_thinking=True,
        long_context_1m=True,
        default_long_context_1m=True,
        sampling=False,
        pricing=ModelPricing(input=10.00, output=50.00, cached_input=1.00),
    ),
    "claude-opus-5": AnthropicModelSpec(
        context_tokens=LONG_CONTEXT_1M_TOKENS,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        # NOTE: thinking is on by default (an omitted `thinking` runs adaptive);
        # `disabled` is only accepted at effort high or lower. Sampling kwargs
        # are rejected as deprecated.
        adaptive_thinking=True,
        long_context_1m=True,
        default_long_context_1m=True,
        sampling=False,
        pricing=ModelPricing(input=5.00, output=25.00, cached_input=0.50),
    ),
    "claude-sonnet-5": AnthropicModelSpec(
        context_tokens=LONG_CONTEXT_1M_TOKENS,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        # NOTE: adaptive-thinking era; sampling kwargs are rejected as deprecated
        # (verified 400 at API_PINNED).
        adaptive_thinking=True,
        long_context_1m=True,
        default_long_context_1m=True,
        sampling=False,
        # NOTE: introductory pricing through 2026-08-31, then $3.00/$15.00.
        pricing=ModelPricing(input=2.00, output=10.00, cached_input=0.20),
    ),
    "claude-opus-4-8": AnthropicModelSpec(
        context_tokens=LONG_CONTEXT_1M_TOKENS,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        adaptive_thinking=True,
        long_context_1m=True,
        default_long_context_1m=True,
        sampling=False,
        pricing=ModelPricing(input=5.00, output=25.00, cached_input=0.50),
    ),
    "claude-opus-4-7": AnthropicModelSpec(
        context_tokens=LONG_CONTEXT_1M_TOKENS,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        adaptive_thinking=True,
        long_context_1m=True,
        default_long_context_1m=True,
        sampling=False,
        pricing=ModelPricing(input=5.00, output=25.00, cached_input=0.50),
    ),
    "claude-opus-4-6": AnthropicModelSpec(
        context_tokens=200_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        adaptive_thinking=True,
        long_context_1m=True,
        default_long_context_1m=False,
        sampling=True,
        pricing=ModelPricing(input=5.00, output=25.00, cached_input=0.50),
    ),
    "claude-sonnet-4-6": AnthropicModelSpec(
        context_tokens=200_000,
        response_tokens=64_000,
        reasoning=True,
        vision=True,
        adaptive_thinking=True,
        long_context_1m=True,
        default_long_context_1m=False,
        sampling=True,
        pricing=ModelPricing(input=3.00, output=15.00, cached_input=0.30),
    ),
    "claude-opus-4-5": AnthropicModelSpec(
        context_tokens=200_000,
        response_tokens=64_000,
        reasoning=True,
        vision=True,
        adaptive_thinking=False,
        long_context_1m=False,
        default_long_context_1m=False,
        sampling=True,
        pricing=ModelPricing(input=5.00, output=25.00, cached_input=0.50),
    ),
    "claude-opus-4-1": AnthropicModelSpec(
        context_tokens=200_000,
        response_tokens=32_000,
        reasoning=True,
        vision=True,
        adaptive_thinking=False,
        long_context_1m=False,
        default_long_context_1m=False,
        sampling=True,
        pricing=ModelPricing(input=15.00, output=75.00, cached_input=1.50),
    ),
    "claude-haiku-4-5": AnthropicModelSpec(
        context_tokens=200_000,
        response_tokens=64_000,
        reasoning=True,
        vision=True,
        adaptive_thinking=False,
        long_context_1m=False,
        default_long_context_1m=False,
        sampling=True,
        pricing=ModelPricing(input=1.00, output=5.00, cached_input=0.10),
    ),
    "claude-sonnet-4-5": AnthropicModelSpec(
        context_tokens=200_000,
        response_tokens=64_000,
        reasoning=True,
        vision=True,
        adaptive_thinking=False,
        long_context_1m=True,
        default_long_context_1m=False,
        sampling=True,
        pricing=ModelPricing(input=3.00, output=15.00, cached_input=0.30),
    ),
}

SUPPORTED_CHAT_MODELS = [
    f"anthropic:{model}" for model, spec in ANTHROPIC_MODEL_SPECS.items() if not spec.reasoning
]
SUPPORTED_REASONING_MODELS = [
    f"anthropic:{model}" for model, spec in ANTHROPIC_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_ANTHROPIC_MODELS = [f"anthropic:{model}" for model in ANTHROPIC_MODEL_SPECS]


def anthropic_strip_prefix(model_name: str) -> str:
    if model_name.startswith("anthropic:"):
        return model_name.removeprefix("anthropic:")
    return model_name


def anthropic_model_spec_for(model: str) -> AnthropicModelSpec:
    model_id = anthropic_strip_prefix(model)
    try:
        return ANTHROPIC_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported Anthropic model: {model}"
        raise ValueError(msg) from e


def resolve_cache_control(cache_control=None):
    """Anthropic prompt caching is on by default (1h TTL); explicit False disables it."""
    selected = CACHE_CONTROL_1H if cache_control is None else cache_control
    if selected is False:
        return None
    return selected


def build_cache_breakpoint_blocks(text: str, cache_control) -> list[dict[str, JsonValue]]:
    """Build Anthropic text blocks from marker-split segments, cache_control on all but last."""
    segments = split_cache_breakpoints(text)
    breakpoint_count = len(segments) - 1
    if breakpoint_count > MAX_CACHE_BREAKPOINTS:
        msg = f"Anthropic supports at most {MAX_CACHE_BREAKPOINTS} cache breakpoints per request."
        raise ValueError(msg)
    if any(segment == "" for segment in segments[:-1]):
        msg = "Anthropic cache breakpoints must follow non-empty text segments."
        raise ValueError(msg)

    blocks: list[dict[str, JsonValue]] = []
    for index, segment in enumerate(segments):
        block: dict[str, JsonValue] = {"type": "text", "text": segment}
        if index < breakpoint_count:
            block["cache_control"] = dict(cache_control)
        blocks.append(block)
    return blocks


class AnthropicMessage(EngineRequestPayload):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, JsonValue]]


class AnthropicPayload(EngineRequestPayload):
    model: str
    messages: list[AnthropicMessage]
    # NOTE: max_tokens is required by the Anthropic API; the engine defaults it to the
    # model's response budget when the user does not provide one.
    max_tokens: int = Field(gt=0)
    system: str | list[dict[str, JsonValue]] | None = None
    thinking: dict[str, JsonValue] | None = None
    output_config: dict[str, JsonValue] | None = None
    stop_sequences: list[str] | None = None
    temperature: float | int | None = Field(default=None, ge=0, le=1)
    top_p: float | int | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    stream: bool | None = None
    metadata: dict[str, JsonValue] | None = None
    tools: list[dict[str, JsonValue]] | None = None
    tool_choice: Literal["none", "auto", "any", "tool"] | dict[str, JsonValue] | None = None
    long_context_1m: bool | None = None
    cache_control: dict[str, JsonValue] | bool | None = None


class AnthropicOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


AnthropicRequest = EngineAPIRequest[
    AnthropicPayload,
    AnthropicOptions,
]


class AnthropicCountTokensPayload(EngineRequestPayload):
    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict[str, JsonValue]] | None = None


AnthropicCountTokensRequest = EngineAPIRequest[
    AnthropicCountTokensPayload,
    AnthropicOptions,
]


class AnthropicUsage(EngineResponsePayload):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicContentBlock(EngineResponsePayload):
    # NOTE: one tolerant block model for all content kinds (text, thinking, tool_use,
    # redacted_thinking, server_tool_use, ...); unknown kinds must not fail parsing.
    type: str
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, JsonValue] | None = None
    cache_control: dict[str, JsonValue] | None = None


class AnthropicResponse(EngineResponsePayload):
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; streamed
    # responses merge usage from message_start and message_delta events.
    id: str | None = None
    type: str | None = None
    role: str | None = None
    content: list[AnthropicContentBlock] = Field(min_length=1)
    model: str | None = None
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage


class AnthropicCountTokensResponse(EngineResponsePayload):
    input_tokens: int
