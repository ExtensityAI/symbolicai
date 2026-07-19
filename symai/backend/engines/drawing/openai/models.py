"""OpenAI Images API wire models.

Locked against https://platform.openai.com/docs/api-reference/images and the legacy
openai-SDK-based engine (openai==2.43.0 resources/images.py) on API_PINNED.

Endpoints:
- POST /v1/images/generations  (JSON; gpt-image-1 always returns b64_json)
- POST /v1/images/edits        (multipart form; image[]/mask file parts)
- POST /v1/images/variations   (multipart form; dall-e-2 only)
"""

from __future__ import annotations

from pydantic import Field

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_IMAGES_GENERATIONS_URL = f"{OPENAI_API_BASE}/images/generations"
OPENAI_IMAGES_EDITS_URL = f"{OPENAI_API_BASE}/images/edits"
OPENAI_IMAGES_VARIATIONS_URL = f"{OPENAI_API_BASE}/images/variations"


class OpenAIImageGenerationsRequest(EngineRequestPayload):
    """JSON body for POST /images/generations."""

    model: str
    prompt: str
    n: int | None = None
    size: str | None = None
    # NOTE: response_format is dall-e-only; gpt-image-* rejects it and always returns
    # b64_json. The engine only sets it for dall-e-3 (legacy engine behavior).
    response_format: str | None = None
    quality: str | None = None
    style: str | None = None
    background: str | None = None
    moderation: str | None = None
    output_format: str | None = None
    output_compression: int | None = None


class OpenAIImageEditsRequest(EngineRequestPayload):
    """Scalar multipart form fields for POST /images/edits (image/mask ride as file parts)."""

    model: str
    prompt: str
    n: int | None = None
    size: str | None = None
    quality: str | None = None


class OpenAIImageVariationsRequest(EngineRequestPayload):
    """Scalar multipart form fields for POST /images/variations (dall-e-2 only)."""

    model: str
    n: int | None = None
    size: str | None = None
    response_format: str | None = None


class OpenAIImage(EngineResponsePayload):
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


class OpenAIImageUsageInputTokensDetails(EngineResponsePayload):
    image_tokens: int | None = None
    text_tokens: int | None = None


class OpenAIImageUsage(EngineResponsePayload):
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_tokens_details: OpenAIImageUsageInputTokensDetails | None = None


class OpenAIImagesResponse(EngineResponsePayload):
    # NOTE: usage is only returned by gpt-image-* models; dall-e responses omit it.
    created: int | None = None
    data: list[OpenAIImage] = Field(min_length=1)
    usage: OpenAIImageUsage | None = None
