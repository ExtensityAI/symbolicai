import logging
from copy import deepcopy

import numpy as np
import openai

from symai.backend.base import Engine
from symai.backend.mixin.openai import OpenAIMixin
from symai.backend.settings import SYMAI_CONFIG
from symai.utils import silence_noisy_loggers

silence_noisy_loggers("openai")

logger = logging.getLogger(__name__)


class EmbeddingEngine(Engine, OpenAIMixin):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        logger = logging.getLogger("openai")
        logger.setLevel(logging.WARNING)
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = api_key or self.config.get("EMBEDDING_ENGINE_API_KEY")
        self.model = model or self.config.get("EMBEDDING_ENGINE_MODEL")
        if self.id() != "embedding":
            return  # do not initialize if not embedding; avoids conflict with llama.cpp check in EngineRepository.register_from_package
        if not self.api_key:
            msg = (
                "OpenAI API key not found. Please set EMBEDDING_ENGINE_API_KEY "
                "in symai.config.json or pass it to the engine."
            )
            raise ValueError(msg)
        self.client = openai.OpenAI(api_key=self.api_key)
        self.max_tokens = self.api_max_context_tokens()
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
            self.client = openai.OpenAI(api_key=self.api_key)
        if "EMBEDDING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["EMBEDDING_ENGINE_MODEL"]

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        args = argument.args
        kwargs = argument.kwargs

        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        except_remedy = kwargs.get("except_remedy")
        new_dim = kwargs.get("new_dim")

        # Validate inputs - OpenAI only supports text
        for item in inp:
            if not isinstance(item, str):
                msg = (
                    f"OpenAI embedding engine only supports text (str) inputs. "
                    f"Received: {type(item).__name__}. "
                    f"For multimodal embeddings, use a model that supports it (e.g., gemini-embedding-2)."
                )
                raise TypeError(msg)

        try:
            res = self.client.embeddings.create(model=self.model, input=inp)
        except Exception as e:
            if except_remedy is None:
                raise
            callback = self.client.embeddings.create
            res = except_remedy(e, inp, callback, self, *args, **kwargs)

        if new_dim:
            mn = min(
                new_dim, self.embedding_dim
            )  # @NOTE: new_dim should be less than or equal to the original embedding dim
            output = [self._normalize_l2(r.embedding[:mn]) for r in res.data]
        else:
            output = [r.embedding for r in res.data]

        metadata = {"raw_output": res}

        return [output], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "EmbeddingEngine does not support processed_input."
        )
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
