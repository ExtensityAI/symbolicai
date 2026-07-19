"""OpenAI Responses API wire models.

Locked against https://developers.openai.com/api/docs/models + /responses endpoint
Pricing: https://developers.openai.com/api/docs/pricing (short-context standard tier)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload
from symai.backend.usage import ModelPricing
from symai.prompts import split_cache_breakpoints

API_PINNED = "2026-07-17"


@dataclass(frozen=True)
class OpenAIModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    pro: bool
    tokenizer: str
    pricing: ModelPricing | None
    # NOTE: explicit prompt cache breakpoints (prompt_cache_breakpoint mode=explicit)
    # exist only on GPT-5.6 models at API_PINNED.
    explicit_cache: bool = False


OPENAI_MODEL_SPECS = {
    "gpt-5.6-sol": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=5.00, output=30.00, cached_input=0.50),
        explicit_cache=True,
    ),
    "gpt-5.6-terra": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=2.50, output=15.00, cached_input=0.25),
        explicit_cache=True,
    ),
    "gpt-5.6-luna": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=1.00, output=6.00, cached_input=0.10),
        explicit_cache=True,
    ),
    "gpt-5.5": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=5.00, output=30.00, cached_input=0.50),
    ),
    "gpt-5.5-pro": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pro=True,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=30.00, output=180.00),
    ),
    "gpt-5.4": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=2.50, output=15.00, cached_input=0.25),
    ),
    "gpt-5.4-pro": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pro=True,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=30.00, output=180.00),
    ),
    "gpt-5.4-mini": OpenAIModelSpec(
        context_tokens=400_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=0.75, output=4.50, cached_input=0.075),
    ),
    "gpt-5.4-nano": OpenAIModelSpec(
        context_tokens=400_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=0.20, output=1.25, cached_input=0.02),
    ),
    "o3-pro": OpenAIModelSpec(
        context_tokens=200_000,
        response_tokens=100_000,
        reasoning=True,
        vision=True,
        pro=True,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=20.00, output=80.00),
    ),
    "o3": OpenAIModelSpec(
        context_tokens=200_000,
        response_tokens=100_000,
        reasoning=True,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=2.00, output=8.00),
    ),
    "gpt-4.1": OpenAIModelSpec(
        context_tokens=1_047_576,
        response_tokens=32_768,
        reasoning=False,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=2.00, output=8.00, cached_input=0.50),
    ),
    "gpt-4.1-mini": OpenAIModelSpec(
        context_tokens=1_047_576,
        response_tokens=32_768,
        reasoning=False,
        vision=True,
        pro=False,
        tokenizer="o200k_base",
        pricing=ModelPricing(input=0.40, output=1.60, cached_input=0.10),
    ),
}

SUPPORTED_CHAT_MODELS = [model for model, spec in OPENAI_MODEL_SPECS.items() if not spec.reasoning]
SUPPORTED_REASONING_MODELS = [model for model, spec in OPENAI_MODEL_SPECS.items() if spec.reasoning]
SUPPORTED_OPENAI_MODELS = [f"openai:{model}" for model in OPENAI_MODEL_SPECS]


def openai_strip_prefix(model_name: str) -> str:
    if model_name.startswith("openai:"):
        return model_name.removeprefix("openai:")
    return model_name


def openai_model_spec_for(model: str) -> OpenAIModelSpec:
    model_id = openai_strip_prefix(model)
    try:
        return OPENAI_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported OpenAI model: {model}"
        raise ValueError(msg) from e


# NOTE: OpenAI allows at most four explicit cache writes per request (GPT-5.6 docs).
MAX_CACHE_BREAKPOINTS = 4


def build_cache_breakpoint_blocks(text: str) -> list[dict[str, JsonValue]]:
    """Build Responses input content blocks from marker-split text segments.

    Every segment except the last gets a ``prompt_cache_breakpoint`` (mode=explicit)
    directive: the marker position is where the cache write happens."""
    segments = split_cache_breakpoints(text)
    breakpoint_count = len(segments) - 1
    if breakpoint_count > MAX_CACHE_BREAKPOINTS:
        msg = (
            f"OpenAI supports at most {MAX_CACHE_BREAKPOINTS} cache breakpoint writes per request."
        )
        raise ValueError(msg)
    if any(segment == "" for segment in segments[:-1]):
        msg = "OpenAI cache breakpoints must follow non-empty text segments."
        raise ValueError(msg)

    blocks: list[dict[str, JsonValue]] = []
    for index, segment in enumerate(segments):
        if segment == "":
            continue
        block: dict[str, JsonValue] = {"type": "input_text", "text": segment}
        if index < breakpoint_count:
            block["prompt_cache_breakpoint"] = {"mode": "explicit"}
        blocks.append(block)
    return blocks


class OpenAIPayload(EngineRequestPayload):
    background: bool | None = None
    context_management: list[dict[str, JsonValue]] | None = None
    conversation: str | dict[str, JsonValue] | None = None
    include: list[str] | None = None
    input: str | list[dict[str, JsonValue]]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)
    metadata: dict[str, str] | None = None
    model: str
    moderation: dict[str, JsonValue] | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    reasoning: dict[str, JsonValue] | None = None
    safety_identifier: str | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    store: bool | None = None
    stream: bool | None = None
    stream_options: dict[Literal["include_obfuscation"], bool] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )
    temperature: float | int | None = Field(default=None, ge=0, le=2)
    text: dict[str, JsonValue] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    tools: list[dict[str, JsonValue]] | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    top_p: float | int | None = Field(default=None, ge=0, le=1)
    truncation: Literal["auto", "disabled"] | None = None
    user: str | None = None


class OpenAIOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


OpenAIRequest = EngineAPIRequest[
    OpenAIPayload,
    OpenAIOptions,
]


class OpenAIInputTokensDetails(EngineResponsePayload):
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None


class OpenAIOutputTokensDetails(EngineResponsePayload):
    reasoning_tokens: int | None = None


class OpenAIUsage(EngineResponsePayload):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: OpenAIInputTokensDetails | None = None
    output_tokens_details: OpenAIOutputTokensDetails | None = None


class OpenAIOutputText(EngineResponsePayload):
    type: Literal["output_text"]
    text: str
    annotations: list[dict[str, JsonValue]] | None = None


class OpenAISummaryText(EngineResponsePayload):
    type: Literal["summary_text"]
    text: str


class OpenAIOutputItem(EngineResponsePayload):
    # NOTE: one tolerant model for all output item kinds. OpenAI keeps adding item
    # types (web_search_call, mcp_call, ...); unknown kinds must not fail parsing.
    type: str
    role: str | None = None
    status: str | None = None
    content: list[OpenAIOutputText] | None = None
    summary: list[OpenAISummaryText] | None = None
    name: str | None = None
    arguments: str | None = None
    call_id: str | None = None


class OpenAIResponse(EngineResponsePayload):
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; Responses
    # streams carry the full response (with usage) in the terminal completed event.
    status: str
    output: list[OpenAIOutputItem]
    usage: OpenAIUsage
    id: str | None = None
    model: str | None = None
    error: dict[str, JsonValue] | None = None
    incomplete_details: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def require_message_content(self):
        # NOTE: message items always carry content in real Responses payloads; its
        # absence means the response shape drifted — fail fast instead of emitting
        # an empty answer silently.
        for item in self.output:
            if item.type == "message" and not item.content:
                msg = "OpenAI response message item requires content."
                raise ValueError(msg)
        return self


def openai_normalize_model(model: str | None) -> str | None:
    """Canonicalize a bare model name to the prefixed form ('o3' -> 'openai:o3').

    DynamicEngine and explicit constructors accept both forms; the wire and the
    supported-model lists use the prefixed form everywhere else.
    """
    if model and ":" not in model:
        return f"openai:{model}"
    return model
