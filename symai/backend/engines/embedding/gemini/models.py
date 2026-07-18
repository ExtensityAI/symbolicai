"""Google Gemini batchEmbedContents wire models for embeddings.

Locked against https://ai.google.dev/api/embeddings#method:-models.batchembedcontents
Verified live at API_PINNED (see engine module docstring for probe notes).
"""

from __future__ import annotations

from pydantic import Field

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# model -> (context tokens, embedding dimensions)
GEMINI_EMBEDDING_MODEL_SPECS = {
    "gemini-embedding-001": (2048, 3072),
    "gemini-embedding-2": (8192, 3072),
}


class GeminiEmbedInlineData(EngineRequestPayload):
    # NOTE: camelCase wire (verified at API_PINNED); `data` is base64-encoded bytes.
    mime_type: str = Field(alias="mimeType")
    data: str


class GeminiEmbedPart(EngineRequestPayload):
    text: str | None = None
    inline_data: GeminiEmbedInlineData | None = Field(default=None, alias="inlineData")


class GeminiEmbedContent(EngineRequestPayload):
    parts: list[GeminiEmbedPart] = Field(min_length=1)


class GeminiEmbedRequestEntry(EngineRequestPayload):
    # NOTE: every entry must repeat the model ("models/<name>") — the batch endpoint
    # does not inherit it from the URL (verified at API_PINNED).
    model: str
    content: GeminiEmbedContent
    task_type: str = Field(default="SEMANTIC_SIMILARITY", alias="taskType")
    # NOTE: outputDimensionality is honored per request entry (verified live at
    # API_PINNED: 3072 -> 768 dims, L2 norm 0.585 for gemini-embedding-001).
    output_dimensionality: int | None = Field(default=None, alias="outputDimensionality")


class GeminiBatchEmbedRequest(EngineRequestPayload):
    requests: list[GeminiEmbedRequestEntry] = Field(min_length=1)


class GeminiEmbedding(EngineResponsePayload):
    values: list[float]


class GeminiBatchEmbedResponse(EngineResponsePayload):
    # NOTE: gemini-embedding-2 additionally returns usageMetadata; it is not consumed
    # (MetadataTracker has no GeminiEmbeddingEngine branch) and stays ignored.
    embeddings: list[GeminiEmbedding] = Field(min_length=1)
