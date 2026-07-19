"""Google Gemini generateContent wire models.

Locked against https://ai.google.dev/gemini-api/docs (REST v1beta, camelCase)
Pricing: https://ai.google.dev/gemini-api/docs/pricing (standard paid tier, <=200K)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload
from symai.backend.usage import ModelPricing

API_PINNED = "2026-07-17"


@dataclass(frozen=True)
class GoogleModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    pricing: ModelPricing | None


GOOGLE_MODEL_SPECS = {
    "gemini-3.1-flash-lite": GoogleModelSpec(
        context_tokens=1_048_576,
        response_tokens=65_536,
        reasoning=False,
        vision=True,
        pricing=ModelPricing(input=0.25, output=1.50, cached_input=0.025),
    ),
    "gemini-3.5-flash": GoogleModelSpec(
        context_tokens=1_048_576,
        response_tokens=65_536,
        reasoning=True,
        vision=True,
        pricing=ModelPricing(input=1.50, output=9.00, cached_input=0.15),
    ),
    "gemini-3.1-pro-preview": GoogleModelSpec(
        context_tokens=1_048_576,
        response_tokens=65_536,
        reasoning=True,
        vision=True,
        pricing=ModelPricing(input=2.00, output=12.00, cached_input=0.20),
    ),
    "gemini-3-flash-preview": GoogleModelSpec(
        context_tokens=1_048_576,
        response_tokens=65_536,
        reasoning=True,
        vision=True,
        pricing=ModelPricing(input=0.50, output=3.00, cached_input=0.05),
    ),
    "gemini-2.5-pro": GoogleModelSpec(
        context_tokens=1_048_576,
        response_tokens=65_536,
        reasoning=True,
        vision=True,
        pricing=ModelPricing(input=1.25, output=10.00, cached_input=0.125),
    ),
    "gemini-2.5-flash": GoogleModelSpec(
        context_tokens=1_048_576,
        response_tokens=65_536,
        reasoning=True,
        vision=True,
        pricing=ModelPricing(input=0.30, output=2.50, cached_input=0.03),
    ),
}

SUPPORTED_CHAT_MODELS = [
    f"gemini:{model}" for model, spec in GOOGLE_MODEL_SPECS.items() if not spec.reasoning
]
SUPPORTED_REASONING_MODELS = [
    f"gemini:{model}" for model, spec in GOOGLE_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_GOOGLE_MODELS = [f"gemini:{model}" for model in GOOGLE_MODEL_SPECS]


def google_strip_prefix(model_name: str) -> str:
    if model_name.startswith("gemini:"):
        return model_name.removeprefix("gemini:")
    return model_name


def google_model_spec_for(model: str) -> GoogleModelSpec:
    model_id = google_strip_prefix(model)
    try:
        return GOOGLE_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported Google model: {model}"
        raise ValueError(msg) from e


class GoogleContent(EngineRequestPayload):
    role: Literal["user", "model"]
    parts: list[dict[str, JsonValue]]


class GoogleSystemInstruction(EngineRequestPayload):
    parts: list[dict[str, JsonValue]]


class GoogleGenerationConfig(EngineRequestPayload):
    max_output_tokens: int | None = Field(default=None, gt=0, alias="maxOutputTokens")
    temperature: float | int | None = None
    top_p: float | int | None = Field(default=None, alias="topP")
    top_k: int | None = Field(default=None, gt=0, alias="topK")
    stop_sequences: list[str] | None = Field(default=None, alias="stopSequences")
    response_mime_type: str | None = Field(default=None, alias="responseMimeType")
    thinking_config: dict[str, JsonValue] | None = Field(default=None, alias="thinkingConfig")


class GoogleTool(EngineRequestPayload):
    function_declarations: list[dict[str, JsonValue]] = Field(alias="functionDeclarations")


class GooglePayload(EngineRequestPayload):
    contents: list[GoogleContent]
    system_instruction: GoogleSystemInstruction | None = Field(
        default=None, alias="systemInstruction"
    )
    generation_config: GoogleGenerationConfig | None = Field(default=None, alias="generationConfig")
    tools: list[GoogleTool] | None = None
    tool_config: dict[str, JsonValue] | None = Field(default=None, alias="toolConfig")


class GoogleOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


GoogleRequest = EngineAPIRequest[
    GooglePayload,
    GoogleOptions,
]


class GoogleCountContent(EngineRequestPayload):
    # NOTE: countTokens tolerates a system role in contents (verified at API_PINNED);
    # generateContent's contents do not.
    role: Literal["user", "model", "system"]
    parts: list[dict[str, JsonValue]]


class GoogleCountTokensPayload(EngineRequestPayload):
    contents: list[GoogleCountContent]


GoogleCountTokensRequest = EngineAPIRequest[
    GoogleCountTokensPayload,
    GoogleOptions,
]


class GoogleUsageMetadata(EngineResponsePayload):
    prompt_token_count: int = Field(alias="promptTokenCount")
    total_token_count: int = Field(alias="totalTokenCount")
    candidates_token_count: int | None = Field(default=None, alias="candidatesTokenCount")
    thoughts_token_count: int | None = Field(default=None, alias="thoughtsTokenCount")
    cached_content_token_count: int | None = Field(default=None, alias="cachedContentTokenCount")


class GooglePart(EngineResponsePayload):
    text: str | None = None
    thought: bool | None = None
    function_call: dict[str, JsonValue] | None = Field(default=None, alias="functionCall")


class GoogleCandidateContent(EngineResponsePayload):
    role: str | None = None
    parts: list[GooglePart] | None = None


class GoogleCandidate(EngineResponsePayload):
    content: GoogleCandidateContent | None = None
    finish_reason: str | None = Field(default=None, alias="finishReason")


class GoogleResponse(EngineResponsePayload):
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; streamed
    # responses carry usageMetadata on the final chunk. Empty parts are valid
    # (budget burn, safety); empty candidates are not.
    candidates: list[GoogleCandidate] = Field(min_length=1)
    usage_metadata: GoogleUsageMetadata = Field(alias="usageMetadata")


class GoogleCountTokensResponse(EngineResponsePayload):
    total_tokens: int = Field(alias="totalTokens")


def google_normalize_model(model: str | None) -> str | None:
    """Canonicalize a bare model name to the prefixed form ('o3' -> 'gemini:o3').

    DynamicEngine and explicit constructors accept both forms; the wire and the
    supported-model lists use the prefixed form everywhere else.
    """
    if model and ":" not in model:
        return f"gemini:{model}"
    return model
