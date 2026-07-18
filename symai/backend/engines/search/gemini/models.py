"""Google Gemini Interactions API wire models for grounded search.

API docs: https://ai.google.dev/gemini-api/docs/interactions
"""

from __future__ import annotations

from typing import Literal

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

SUPPORTED_SEARCH_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
]
DEFAULT_SEARCH_MODEL = "gemini-3.5-flash"


class GeminiInteractionTool(EngineRequestPayload):
    type: Literal["google_search"]


class GeminiInteractionRequest(EngineRequestPayload):
    # NOTE: snake_case wire (verified at API_PINNED); input is a plain string.
    model: str
    input: str
    tools: list[GeminiInteractionTool]
    system_instruction: str | None = None


class GeminiInteractionAnnotation(EngineResponsePayload):
    type: str | None = None
    url: str | None = None
    title: str | None = None
    start_index: int | None = None
    end_index: int | None = None


class GeminiInteractionContent(EngineResponsePayload):
    # NOTE: content block shapes differ per step type (google_search_call carries query
    # blocks, model_output carries text + inline annotations), so every field is optional.
    type: str | None = None
    text: str | None = None
    annotations: list[GeminiInteractionAnnotation] | None = None


class GeminiInteractionStep(EngineResponsePayload):
    # NOTE: observed step types at API_PINNED: google_search_call, google_search_result,
    # thought, model_output.
    type: str
    content: list[GeminiInteractionContent] | None = None


class GeminiGroundingToolCount(EngineResponsePayload):
    type: str | None = None
    count: int | None = None


class GeminiInteractionUsage(EngineResponsePayload):
    # NOTE: MetadataTracker's GeminiSearchEngine branch adds total_input_tokens /
    # total_output_tokens / total_tokens unguarded, so they default to 0; the
    # cached/thought fields and grounding_tool_count are guarded there with `or 0`,
    # so they stay optional (verified present in live usage at API_PINNED).
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int | None = None
    total_thought_tokens: int | None = None
    grounding_tool_count: list[GeminiGroundingToolCount] | None = None


class GeminiInteractionResponse(EngineResponsePayload):
    steps: list[GeminiInteractionStep] | None = None
    usage: GeminiInteractionUsage | None = None
    # NOTE: convenience text field the SDK surfaced; kept so GeminiSearchResult's
    # output_text fallback keeps working when it appears on the wire.
    output_text: str | None = None
