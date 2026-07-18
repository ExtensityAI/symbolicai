"""OpenAI Responses API wire models for the web_search tool.

Locked against https://platform.openai.com/docs/api-reference/responses/create
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_RESPONSES_URL = f"{OPENAI_API_BASE}/responses"


class OpenAISearchTool(EngineRequestPayload):
    type: Literal["web_search"] = "web_search"
    # NOTE: user_location is the provider's approximate-location object
    # ({"type": "approximate", "country"?, "city"?, "region"?, "timezone"?});
    # filters carries {"allowed_domains": [...]} — both are free-form vendor objects.
    user_location: dict[str, JsonValue] | None = None
    filters: dict[str, JsonValue] | None = None


class OpenAISearchRequestPayload(EngineRequestPayload):
    model: str
    input: list[dict[str, JsonValue]] = Field(min_length=1)
    tools: list[OpenAISearchTool] = Field(min_length=1)
    # NOTE: "auto" lets reasoning models decide when to search; non-reasoning models get
    # the forced {"type": "web_search"} form so a citation-bearing answer is guaranteed.
    tool_choice: dict[str, str] | str | None = None
    reasoning: dict[str, JsonValue] | None = None


class OpenAISearchUrlCitation(EngineResponsePayload):
    type: str
    url: str | None = None
    title: str | None = None
    start_index: int = 0
    end_index: int = 0


class OpenAISearchContent(EngineResponsePayload):
    type: str
    text: str | None = None
    annotations: list[OpenAISearchUrlCitation] | None = None


class OpenAISearchOutputItem(EngineResponsePayload):
    # NOTE: output[] also carries non-message items (web_search_call, reasoning, ...);
    # only "message" items bear content — the rest parse with content=None.
    type: str
    content: list[OpenAISearchContent] | None = None


class OpenAISearchInputTokensDetails(EngineResponsePayload):
    cached_tokens: int = 0


class OpenAISearchOutputTokensDetails(EngineResponsePayload):
    reasoning_tokens: int = 0


class OpenAISearchUsage(EngineResponsePayload):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # NOTE: MetadataTracker reads the nested details unconditionally; default them so a
    # response that omits the details objects still tracks as zero, not AttributeError.
    input_tokens_details: OpenAISearchInputTokensDetails = Field(
        default_factory=OpenAISearchInputTokensDetails
    )
    output_tokens_details: OpenAISearchOutputTokensDetails = Field(
        default_factory=OpenAISearchOutputTokensDetails
    )


class OpenAISearchResponse(EngineResponsePayload):
    output: list[OpenAISearchOutputItem] = Field(default_factory=list)
    output_text: str | None = None
    usage: OpenAISearchUsage | None = None
    error: dict[str, JsonValue] | None = None
