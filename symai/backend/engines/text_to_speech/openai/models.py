"""OpenAI Audio Speech API wire models.

Locked against https://platform.openai.com/docs/api-reference/audio/createSpeech

The success response is the raw audio file content (bytes, not JSON) whose
content type follows response_format; only the error envelope is JSON (handled
by the shared transport error lattice).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from symai.backend.request import EngineRequestPayload

API_PINNED = "2026-07-18"

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_SPEECH_URL = f"{OPENAI_API_BASE}/audio/speech"

# NOTE: response_format -> expected success content type; mp3 is the provider default
# when response_format is omitted from the wire payload.
OPENAI_SPEECH_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

OpenAISpeechFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class OpenAISpeechRequest(EngineRequestPayload):
    model: str
    input: str = Field(min_length=1, max_length=4096)
    voice: str = Field(min_length=1)
    response_format: OpenAISpeechFormat | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
