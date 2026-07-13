import logging
from copy import deepcopy

import httpx
import numpy as np
from pydantic import TypeAdapter

from symai.backend.base import Engine
from symai.backend.integrations.openai.client import Client as OpenAIClient
from symai.backend.integrations.openai.embeddings import (
    CreateEmbeddingRequest,
    EmbeddingList,
    EmbeddingModel,
)
from symai.backend.integrations.openai.transport import APIResponse
from symai.backend.mixin.openai import OpenAIMixin
from symai.backend.settings import SYMAI_CONFIG

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_ADAPTER = TypeAdapter(EmbeddingModel)


class OpenAIEmbeddingEngine(Engine, OpenAIMixin):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client_timeout: float | None = None,
        http_client: httpx.Client | None = None,
    ):
        super().__init__(client_timeout=client_timeout)
        self.config = deepcopy(SYMAI_CONFIG)
        configured_api_key = api_key or self.config.get("EMBEDDING_ENGINE_API_KEY")
        configured_model = model or self.config.get("EMBEDDING_ENGINE_MODEL")
        self.api_key = configured_api_key if isinstance(configured_api_key, str) else ""
        self.model = configured_model if isinstance(configured_model, str) else ""
        if self.id() != "embedding":
            return
        if not self.api_key:
            msg = (
                "OpenAI API key not found. Please set EMBEDDING_ENGINE_API_KEY "
                "in symai.config.json or pass it to the engine."
            )
            raise ValueError(msg)
        self.http_client = http_client
        self.max_tokens = self.api_embedding_context_tokens()
        self.embedding_dim = self.api_embedding_dims()
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.api_key and self.model and self.model.startswith("text-embedding"):
            return "embedding"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "EMBEDDING_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["EMBEDDING_ENGINE_API_KEY"]
        if "EMBEDDING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["EMBEDDING_ENGINE_MODEL"]

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
            model=EMBEDDING_MODEL_ADAPTER.validate_python(self.model),
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
        if self.http_client is not None:
            return OpenAIClient(
                api_key=self.api_key,
                http_client=self.http_client,
            ).create_embeddings(request)

        with httpx.Client(timeout=self.client_timeout) as http_client:
            return OpenAIClient(
                api_key=self.api_key,
                http_client=http_client,
            ).create_embeddings(request)

    def prepare(self, argument):
        if argument.prop.processed_input:
            msg = "OpenAIEmbeddingEngine does not support processed_input."
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
