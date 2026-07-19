"""Google Gemini generateContent wire models for image generation.

Locked against https://ai.google.dev/api/generate-content and
https://ai.google.dev/gemini-api/docs/image-generation.
Verified live at API_PINNED (see engine module for usage).
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

SUPPORTED_IMAGE_MODEL_PREFIXES = ("gemini-2.5-flash-image", "gemini-3-pro-image-preview")


class GeminiImagePart(EngineRequestPayload):
    text: str


class GeminiImageContent(EngineRequestPayload):
    parts: list[GeminiImagePart] = Field(min_length=1)


class GeminiImageGenerationConfig(EngineRequestPayload):
    # NOTE: camelCase wire (verified at API_PINNED); IMAGE-only requests suppress the
    # conversational text parts the model emits by default. Extra keys are allowed so
    # callers can pass additional wire-format generationConfig fields through
    # (the legacy kwargs['config'] GenerateContentConfig pass-through, as a plain dict).
    model_config = ConfigDict(extra="allow", strict=True, populate_by_name=True)

    response_modalities: list[str] = Field(alias="responseModalities", min_length=1)


class GeminiImageGenerateRequest(EngineRequestPayload):
    contents: list[GeminiImageContent] = Field(min_length=1)
    generation_config: GeminiImageGenerationConfig = Field(alias="generationConfig")


class GeminiImageInlineData(EngineResponsePayload):
    # NOTE: REST wire is camelCase inlineData/mimeType (the google.genai SDK mapped them
    # to inline_data/mime_type); `data` is base64-encoded image bytes.
    mime_type: str | None = Field(default=None, alias="mimeType")
    data: str | None = None


class GeminiImageResponsePart(EngineResponsePayload):
    # NOTE: image responses may still carry text parts alongside inlineData parts, so
    # every field stays optional; the result filters on inline_data presence.
    text: str | None = None
    inline_data: GeminiImageInlineData | None = Field(default=None, alias="inlineData")


class GeminiImageCandidateContent(EngineResponsePayload):
    parts: list[GeminiImageResponsePart] | None = None


class GeminiImageCandidate(EngineResponsePayload):
    content: GeminiImageCandidateContent | None = None


class GeminiImageGenerateResponse(EngineResponsePayload):
    # NOTE: promptFeedback/usageMetadata are returned but not consumed and stay ignored.
    candidates: list[GeminiImageCandidate] = Field(min_length=1)
