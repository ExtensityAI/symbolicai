"""llama.cpp embedding wire models (local symserver).

The server speaks its own embedding shape (not the OpenAI-compatible one):
POST /v1/embeddings {"content": str | list[str], "embd_normalize": int} and
answers a bare list of {"index": int, "embedding": [[...]]} objects, one per
input. Current llama.cpp builds nest the vector one level (one inner vector
per input sequence — exactly one for string content); older builds answered
with a flat {"embedding": [...]}, which the item validator wraps into the
nested shape so both server generations parse.
"""

from pydantic import JsonValue, field_validator

from symai.backend.request import EngineResponsePayload

API_PINNED = "2026-07-18"


class LlamaCppEmbeddingItem(EngineResponsePayload):
    index: int | None = None
    # One inner vector per input sequence; string content yields exactly one.
    embedding: list[list[float]]

    @field_validator("embedding", mode="before")
    @classmethod
    def _wrap_legacy_flat_vector(cls, value):
        # Older llama.cpp builds answered {"embedding": [floats]} with no nesting.
        if isinstance(value, list) and (not value or isinstance(value[0], (int, float))):
            return [value]
        return value


# NOTE: the wire answer is a top-level JSON array, not an object.
LlamaCppEmbeddingResponse = list[LlamaCppEmbeddingItem]


class LlamaCppEmbeddingError(EngineResponsePayload):
    error: JsonValue | None = None
