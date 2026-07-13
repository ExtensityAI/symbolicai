from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from symai.clients._models import ModelId, StrictModel, TolerantModel

PATH = "/embeddings"


Model = Literal[
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    context_tokens: int
    dimensions: int


MODEL_SPECS: dict[Model, ModelSpec] = {
    "text-embedding-ada-002": ModelSpec(8_191, 1_536),
    "text-embedding-3-small": ModelSpec(8_191, 1_536),
    "text-embedding-3-large": ModelSpec(8_191, 3_072),
}


class CreateEmbeddingRequest(StrictModel):
    input: str | tuple[str, ...]
    model: Model | ModelId
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
    model: str
    usage: Usage
