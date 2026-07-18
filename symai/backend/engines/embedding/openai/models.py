"""OpenAI Embeddings API wire models.

Locked against https://platform.openai.com/docs/api-reference/embeddings/create
"""

from __future__ import annotations

from pydantic import Field

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_EMBEDDINGS_URL = f"{OPENAI_API_BASE}/embeddings"

# model -> (context tokens, embedding dimensions)
OPENAI_EMBEDDING_MODEL_SPECS = {
    "text-embedding-ada-002": (8192, 1536),
    "text-embedding-3-small": (8192, 1536),
    "text-embedding-3-large": (8192, 3072),
}


class OpenAIEmbeddingRequest(EngineRequestPayload):
    # NOTE: `dimensions` is intentionally NOT sent — new_dim truncation is client-side
    # (L2 re-normalized), so the wire payload is exactly {model, input}.
    model: str
    input: list[str] = Field(min_length=1)


class OpenAIEmbeddingData(EngineResponsePayload):
    embedding: list[float]
    index: int


class OpenAIEmbeddingUsage(EngineResponsePayload):
    # NOTE: components.py MetadataTracker's "EmbeddingEngine" branch reads
    # metadata["raw_output"].usage.prompt_tokens / .total_tokens — keep both required.
    prompt_tokens: int
    total_tokens: int


class OpenAIEmbeddingResponse(EngineResponsePayload):
    data: list[OpenAIEmbeddingData] = Field(min_length=1)
    usage: OpenAIEmbeddingUsage
