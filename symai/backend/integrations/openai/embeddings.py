from typing import Literal

from pydantic import Field

from symai.backend.integrations.base import StrictModel, TolerantModel

PATH = "/embeddings"


class EmbeddingRequest(StrictModel):
    input: str | tuple[str, ...]
    model: str
    dimensions: int | None = Field(default=None, gt=0)
    encoding_format: Literal["float", "base64"] | None = None
    user: str | None = None


class EmbeddingData(TolerantModel):
    object: str | None = None
    embedding: tuple[float, ...] | str
    index: int


class Usage(TolerantModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(TolerantModel):
    object: str | None = None
    data: tuple[EmbeddingData, ...]
    model: str | None = None
    usage: Usage
