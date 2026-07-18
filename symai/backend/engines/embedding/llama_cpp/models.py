"""llama.cpp embedding wire models (local symserver).

The server speaks its own embedding shape (not the OpenAI-compatible one):
POST /v1/embeddings {"content": str | list[str], "embd_normalize": int} and
answers a bare list of {"embedding": [...]} objects, one per input.
"""

from pydantic import JsonValue

from symai.backend.request import EngineResponsePayload

API_PINNED = "2026-07-18"


class LlamaCppEmbeddingItem(EngineResponsePayload):
    embedding: list[float]


# NOTE: the wire answer is a top-level JSON array, not an object.
LlamaCppEmbeddingResponse = list[LlamaCppEmbeddingItem]


class LlamaCppEmbeddingError(EngineResponsePayload):
    error: JsonValue | None = None
