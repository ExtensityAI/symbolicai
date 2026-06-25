import logging
from copy import deepcopy

import filetype
import numpy as np
from google import genai
from google.genai.types import Content, EmbedContentConfig, Part

from ...base import Engine
from ...mixin.google import GoogleMixin
from ...settings import SYMAI_CONFIG

logging.getLogger("google.generativeai").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class GeminiEmbeddingEngine(Engine, GoogleMixin):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = api_key or self.config.get("EMBEDDING_ENGINE_API_KEY")
        self.model = model or self.config.get("EMBEDDING_ENGINE_MODEL", "gemini-embedding-001")
        if self.id() != "embedding":
            return
        if not self.api_key:
            msg = (
                "Gemini API key not found. Please set EMBEDDING_ENGINE_API_KEY "
                "in symai.config.json or pass it to the engine."
            )
            raise ValueError(msg)

        self.client = genai.Client(api_key=self.api_key)
        self.name = self.__class__.__name__
        self.embedding_dim = self.api_embedding_dims()
        self.max_tokens = self.api_max_context_tokens()

    def id(self) -> str:
        if self.api_key and self.model and self.model.startswith("gemini"):
            return "embedding"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "EMBEDDING_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["EMBEDDING_ENGINE_API_KEY"]
            self.client = genai.Client(api_key=self.api_key)
        if "EMBEDDING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["EMBEDDING_ENGINE_MODEL"]

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        kwargs = argument.kwargs

        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        new_dim = kwargs.get("new_dim")
        task_type = kwargs.get("task_type", "SEMANTIC_SIMILARITY")

        # Convert inputs to Google genai format
        # Supports: str, bytes, Part, Content
        converted_inp = []
        for item in inp:
            if isinstance(item, str):  # Text input - pass through
                converted_inp.append(item)
            elif isinstance(item, bytes):  # Raw bytes - detect mime type and convert to Part
                # NOTE: filetype is a lightweight dependency (~50KB) that auto-detects MIME types
                # from raw bytes. This improves user experience by allowing Symbol(bytes).embed()
                # without requiring explicit Part construction with mime_type.
                mime_type = filetype.guess_mime(item) or "application/octet-stream"
                converted_inp.append(Part.from_bytes(data=item, mime_type=mime_type))
            elif isinstance(item, (Part, Content)):  # Already in Google format - pass through
                converted_inp.append(item)
            else:  # Fallback: convert to string
                converted_inp.append(str(item))

        inp = converted_inp

        config = EmbedContentConfig(task_type=task_type)
        if new_dim is not None:
            config.output_dimensionality = new_dim

        res = self.client.models.embed_content(
            model=self.model,
            contents=inp,
            config=config,
        )

        output = [emb.values for emb in res.embeddings]

        if output and isinstance(output[0], (list, np.ndarray)):
            self.embedding_dim = len(output[0])

        # NOTE: Confirmed empirically: gemini-embedding-001 returns ||v||=0.585 at dim=768
        # (requires client-side L2 normalization); gemini-embedding-2 returns ||v||=1.0 at
        # dim=768 (auto-normalized server-side). Re-normalizing a unit vector is idempotent
        # (v / ||v|| = v when ||v|| = 1), so this is safe for both models.
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
