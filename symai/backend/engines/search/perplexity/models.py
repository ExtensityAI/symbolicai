"""Perplexity chat completions wire models.

Locked against https://docs.perplexity.ai/api-reference/chat-completions-post
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

PERPLEXITY_CHAT_COMPLETIONS_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityMessage(EngineRequestPayload):
    role: Literal["system", "user", "assistant"]
    content: str


class PerplexityRequestPayload(EngineRequestPayload):
    model: str
    messages: list[PerplexityMessage] = Field(min_length=1)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_format: dict[str, JsonValue] | None = None
    search_domain_filter: list[str] | None = None
    return_images: bool | None = None
    return_related_questions: bool | None = None
    search_recency_filter: Literal["hour", "day", "week", "month"] | None = None
    web_search_options: dict[str, JsonValue] | None = None


class PerplexityResponseMessage(EngineResponsePayload):
    role: str
    content: str


class PerplexityChoice(EngineResponsePayload):
    index: int
    message: PerplexityResponseMessage
    finish_reason: str | None = None


class PerplexityUsage(EngineResponsePayload):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class PerplexityResponse(EngineResponsePayload):
    choices: list[PerplexityChoice]
    citations: list[str] | None = None
    usage: PerplexityUsage | None = None
