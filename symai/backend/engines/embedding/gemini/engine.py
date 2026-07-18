from __future__ import annotations

import base64
import logging
from copy import deepcopy

import filetype
import numpy as np

from symai.backend.base import Engine
from symai.backend.engines.embedding.gemini.models import (
    GEMINI_API_BASE,
    GEMINI_EMBEDDING_MODEL_SPECS,
    GeminiBatchEmbedRequest,
    GeminiBatchEmbedResponse,
    GeminiEmbedContent,
    GeminiEmbedInlineData,
    GeminiEmbedPart,
    GeminiEmbedRequestEntry,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import DEFAULT_RETRIES, execute_engine_api_request

logger = logging.getLogger(__name__)


class GeminiEmbeddingEngine(Engine):
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

        self.name = self.__class__.__name__
        self.embedding_dim = self.api_embedding_dims()
        self.max_tokens = self.api_max_context_tokens()
        self.transport_client = None

    def id(self) -> str:
        if self.api_key and self.model and self.model.startswith("gemini"):
            return "embedding"
        return super().id()  # default to unregistered

    def api_max_context_tokens(self) -> int:
        return GEMINI_EMBEDDING_MODEL_SPECS[self.model][0]

    def api_embedding_dims(self) -> int:
        return GEMINI_EMBEDDING_MODEL_SPECS[self.model][1]

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "EMBEDDING_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["EMBEDDING_ENGINE_API_KEY"]
            # NOTE: the shared transport is stateless (headers are built per request from
            # self.api_key), so a key change needs no client rebuild.
        if "EMBEDDING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["EMBEDDING_ENGINE_MODEL"]

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def build_request(self, argument) -> EngineAPIRequest:
        prepared_input = argument.prop.prepared_input
        kwargs = argument.kwargs

        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        new_dim = kwargs.get("new_dim")
        task_type = kwargs.get("task_type", "SEMANTIC_SIMILARITY")
        # NOTE: new_dim steers response post-processing (client-side L2 truncate +
        # normalize), not the typed response; stash it for parse_response, which only
        # receives the response.
        self._new_dim = new_dim

        # NOTE: The batch endpoint embeds every request entry separately, so a list
        # input becomes one entry per item (matching the old SDK-based engine, which
        # wrapped each item in its own Content).
        entries = [
            GeminiEmbedRequestEntry(
                model=f"models/{self.model}",
                content=self._to_content(item),
                task_type=task_type,
                output_dimensionality=new_dim,
            )
            for item in inp
        ]
        payload = GeminiBatchEmbedRequest(requests=entries)
        return EngineAPIRequest(
            provider="google",
            operation="batchEmbedContents",
            payload=payload,
            method="POST",
            url=f"{GEMINI_API_BASE}/models/{self.model}:batchEmbedContents",
            headers={"x-goog-api-key": self.api_key},
            timeout=self.client_timeout,
        )

    def call_request(self, request: EngineAPIRequest) -> GeminiBatchEmbedResponse:
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        return GeminiBatchEmbedResponse.model_validate(response.json())

    def parse_response(self, response: GeminiBatchEmbedResponse):
        output = [emb.values for emb in response.embeddings]

        if output and isinstance(output[0], list):
            self.embedding_dim = len(output[0])

        # NOTE: Confirmed empirically: gemini-embedding-001 returns ||v||=0.585 at dim=768
        # (requires client-side L2 normalization); gemini-embedding-2 returns ||v||=1.0 at
        # dim=768 (auto-normalized server-side). Re-normalizing a unit vector is idempotent
        # (v / ||v|| = v when ||v|| = 1), so this is safe for both models.
        new_dim = self._new_dim
        if new_dim:
            mn = min(new_dim, self.embedding_dim)
            output = [self._normalize_l2(emb[:mn]) for emb in output]

        metadata = {"raw_output": response}

        return [output], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "EmbeddingEngine does not support processed_input."
        )
        argument.prop.prepared_input = argument.prop.entries

    def _to_content(self, item) -> GeminiEmbedContent:
        if isinstance(item, str):  # Text input
            return GeminiEmbedContent(parts=[GeminiEmbedPart(text=item)])
        if isinstance(item, bytes):  # Raw bytes - detect mime type and inline as base64
            # NOTE: filetype is a lightweight dependency (~50KB) that auto-detects MIME types
            # from raw bytes. This improves user experience by allowing Symbol(bytes).embed()
            # without requiring explicit Part construction with mime_type.
            mime_type = filetype.guess_mime(item) or "application/octet-stream"
            return GeminiEmbedContent(
                parts=[
                    GeminiEmbedPart(
                        inline_data=GeminiEmbedInlineData(
                            mime_type=mime_type, data=base64.b64encode(item).decode("ascii")
                        )
                    )
                ]
            )
        msg = (
            f"GeminiEmbeddingEngine supports str and bytes inputs; got {type(item).__name__}. "
            "Pass raw file bytes for multimodal inputs (the mime type is auto-detected)."
        )
        raise TypeError(msg)

    def _normalize_l2(self, x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x.tolist()
            return (x / norm).tolist()
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm).tolist()
