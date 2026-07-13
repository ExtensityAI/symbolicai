from dataclasses import dataclass

import numpy as np

from symai.backend.base import Engine
from symai.backend.usage import EngineUsageRecord
from symai.clients.openai.client import Client as OpenAIClient
from symai.clients.openai.embeddings import (
    CreateEmbeddingRequest,
    EmbeddingList,
    EmbeddingModel,
)
from symai.clients.openai.transport import APIResponse


@dataclass(frozen=True, slots=True)
class ModelSpec:
    context_tokens: int
    dimensions: int


MODEL_SPECS: dict[EmbeddingModel, ModelSpec] = {
    "text-embedding-ada-002": ModelSpec(8_191, 1_536),
    "text-embedding-3-small": ModelSpec(8_191, 1_536),
    "text-embedding-3-large": ModelSpec(8_191, 3_072),
}
SUPPORTED_MODELS = tuple(MODEL_SPECS)


class EmbeddingEngine(Engine):
    provider = "openai"
    capability = "embedding"

    def __init__(self, *, client: OpenAIClient, model: EmbeddingModel):
        super().__init__()
        try:
            self.model_spec = MODEL_SPECS[model]
        except KeyError as e:
            msg = f"Unsupported model: {model}"
            raise ValueError(msg) from e

        self.client = client
        self.model = model
        self.max_tokens = self.model_spec.context_tokens
        self.embedding_dim = self.model_spec.dimensions
        self.name = self.__class__.__name__

    def id(self) -> str:
        return "embedding"

    def forward(self, argument):  # pyright: ignore[reportIncompatibleMethodOverride]
        prepared_input = argument.prop.prepared_input
        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        new_dim = argument.kwargs.get("new_dim")

        for item in inp:
            if not isinstance(item, str):
                msg = (
                    "OpenAI embedding engine only supports text (str) inputs. "
                    f"Received: {type(item).__name__}. "
                    "For multimodal embeddings, use a model that supports it."
                )
                raise TypeError(msg)

        request = CreateEmbeddingRequest(
            input=tuple(inp),
            model=self.model,
        )
        response = self.call_request(request)
        raw_output = response.data
        embeddings = []
        for item in raw_output.data:
            if isinstance(item.embedding, str):
                msg = "OpenAI returned a base64 embedding when float encoding was requested."
                raise ValueError(msg)
            embeddings.append(item.embedding)
        if new_dim:
            dimension = min(new_dim, self.embedding_dim)
            output = [self._normalize_l2(embedding[:dimension]) for embedding in embeddings]
        else:
            output = [list(embedding) for embedding in embeddings]

        metadata = {
            "raw_output": raw_output,
            "response": response,
        }
        return [output], metadata

    def call_request(self, request: CreateEmbeddingRequest) -> APIResponse[EmbeddingList]:
        return self.client.create_embeddings(request)

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord:
        usage = metadata["raw_output"].usage
        return EngineUsageRecord(
            prompt_tokens=usage.prompt_tokens,
            total_tokens=usage.total_tokens,
        )

    def prepare(self, argument):
        if argument.prop.processed_input:
            msg = "EmbeddingEngine does not support processed_input."
            raise ValueError(msg)
        argument.prop.prepared_input = argument.prop.entries

    def _normalize_l2(self, x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x.tolist()
            return (x / norm).tolist()
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm).tolist()
