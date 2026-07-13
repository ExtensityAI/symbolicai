from typing import Literal

from pydantic import Field

from symai.clients._models import StrictModel, TolerantModel

PATH = "/embeddings"


EmbeddingModel = Literal[
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
]


class CreateEmbeddingRequest(StrictModel):
    input: str | tuple[str, ...]
    model: EmbeddingModel
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


class EmbeddingList(TolerantModel):
    object: str | None = None
    data: tuple[EmbeddingData, ...]
    model: EmbeddingModel
    usage: Usage
