import logging
import numpy as np

from google import genai
from google.genai.types import EmbedContentConfig

from ...base import Engine
from ...mixin.google import GoogleMixin
from ...settings import SYMAI_CONFIG

logging.getLogger("google.generativeai").setLevel(logging.ERROR)


class GeminiEmbeddingEngine(Engine, GoogleMixin):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        self._api_key = api_key or self.config.get("EMBEDDING_ENGINE_API_KEY")
        self._model = model or self.config.get("EMBEDDING_ENGINE_MODEL", "gemini-embedding-001")
        if self.id() != "embedding":
            return
        if not self._api_key:
            raise ValueError("Gemini API key not found. Please set EMBEDDING_ENGINE_API_KEY in symai.config.json or pass it to the engine.")

        self._client = genai.Client(api_key=self._api_key)

    def id(self) -> str:
        if self._api_key and self._model and "embedding" in self._model:
            return "embedding"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "EMBEDDING_ENGINE_API_KEY" in kwargs:
            self._api_key = kwargs["EMBEDDING_ENGINE_API_KEY"]
            self._client = genai.Client(api_key=self._api_key)
        if "EMBEDDING_ENGINE_MODEL" in kwargs:
            self._model = kwargs["EMBEDDING_ENGINE_MODEL"]

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        # args = argument.args
        kwargs = argument.kwargs

        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        # except_remedy = kwargs.get("except_remedy")
        new_dim = kwargs.get("new_dim")
        task_type = kwargs.get("task_type", "SEMANTIC_SIMILARITY")

        try:
            config = EmbedContentConfig(
                task_type=task_type,
            )
            # Use output_dimensionality in config if supported and new_dim is specified
            if new_dim is not None:
                config.output_dimensionality = new_dim

            res = self._client.models.embed_content(
                model=self._model,
                contents=inp,
                config=config,
            )
        except Exception as e:
            raise e

        output = []
        if hasattr(res, "embeddings") and res.embeddings:
            output = [emb.values for emb in res.embeddings if emb is not None and emb.values is not None]
        elif hasattr(res, "embedding"):
            val = getattr(res, "embedding", None)
            if val is not None:
                if hasattr(val, "values") and val.values is not None:
                    output = [val.values]
                else:
                    output = [val]

        if output and len(output) > 0:
            first_element = output[0]
            if isinstance(first_element, (list, np.ndarray)):
                self.embedding_dim = len(first_element)

        # Apply client-side truncation and l2 normalization ONLY if a new dimension is requested
        if new_dim:
            mn = min(new_dim, self.embedding_dim)
            output = [self._normalize_l2(emb[:mn]) for emb in output]

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
