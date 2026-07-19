from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np

from symai.backend.base import Engine
from symai.backend.engines.embedding.openai.models import (
    OPENAI_EMBEDDING_MODEL_SPECS,
    OPENAI_EMBEDDINGS_URL,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import DEFAULT_RETRIES, execute_engine_api_request
from symai.utils import silence_noisy_loggers

silence_noisy_loggers("openai")

logger = logging.getLogger(__name__)


class EmbeddingEngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        logger_ = logging.getLogger("openai")
        logger_.setLevel(logging.WARNING)
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = api_key or self.config.get("EMBEDDING_ENGINE_API_KEY")
        self.model = model or self.config.get("EMBEDDING_ENGINE_MODEL")
        self.transport_client = None
        if self.id() != "embedding":
            return  # do not initialize if not embedding; avoids conflict with llama.cpp check in EngineRepository.register_from_package
        if not self.api_key:
            msg = (
                "OpenAI API key not found. Please set EMBEDDING_ENGINE_API_KEY "
                "in symai.config.json or pass it to the engine."
            )
            raise ValueError(msg)
        self.max_tokens = self.api_max_context_tokens()
        self.embedding_dim = self.api_embedding_dims()
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.api_key and self.model and self.model.startswith("text-embedding"):
            return "embedding"
        return super().id()

    def api_max_context_tokens(self) -> int:
        return OPENAI_EMBEDDING_MODEL_SPECS[self.model][0]

    def api_embedding_dims(self) -> int:
        return OPENAI_EMBEDDING_MODEL_SPECS[self.model][1]

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "EMBEDDING_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["EMBEDDING_ENGINE_API_KEY"]
            # NOTE: auth headers are built per request on the shared transport, so a key
            # change only needs the cached transport handle dropped, not a client rebuild.
            self.transport_client = None
        if "EMBEDDING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["EMBEDDING_ENGINE_MODEL"]

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response, argument)

    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "EmbeddingEngine does not support processed_input."
        )
        argument.prop.prepared_input = argument.prop.entries

    def build_request(self, argument) -> EngineAPIRequest:
        prepared_input = argument.prop.prepared_input
        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]

        # Validate inputs - OpenAI only supports text
        for item in inp:
            if not isinstance(item, str):
                msg = (
                    f"OpenAI embedding engine only supports text (str) inputs. "
                    f"Received: {type(item).__name__}. "
                    f"For multimodal embeddings, use a model that supports it (e.g., gemini-embedding-2)."
                )
                raise TypeError(msg)

        payload = OpenAIEmbeddingRequest(model=self.model, input=inp)
        return EngineAPIRequest(
            provider="openai",
            operation="embeddings.create",
            payload=payload,
            method="POST",
            url=OPENAI_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.client_timeout,
        )

    def call_request(self, request: EngineAPIRequest) -> OpenAIEmbeddingResponse:
        return self._execute(request)

    def _execute(self, request: EngineAPIRequest) -> OpenAIEmbeddingResponse:
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        return OpenAIEmbeddingResponse.model_validate(response.json())

    def parse_response(self, response: OpenAIEmbeddingResponse, argument):
        new_dim = argument.kwargs.get("new_dim")
        if new_dim:
            mn = min(
                new_dim, self.embedding_dim
            )  # @NOTE: new_dim should be less than or equal to the original embedding dim
            output = [self._normalize_l2(r.embedding[:mn]) for r in response.data]
        else:
            output = [r.embedding for r in response.data]

        metadata = {"raw_output": response}

        return [output], metadata

    def _normalize_l2(self, x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x.tolist()
            return (x / norm).tolist()
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm).tolist()
