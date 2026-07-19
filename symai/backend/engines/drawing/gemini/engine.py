from __future__ import annotations

import base64
import logging
import mimetypes
import tempfile
from pathlib import Path

from symai.backend.base import Engine
from symai.backend.engines.drawing.gemini.models import (
    GEMINI_API_BASE,
    SUPPORTED_IMAGE_MODEL_PREFIXES,
    GeminiImageContent,
    GeminiImageGenerateRequest,
    GeminiImageGenerateResponse,
    GeminiImageGenerationConfig,
    GeminiImagePart,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import DEFAULT_RETRIES, execute_engine_api_request
from symai.symbol import Result

logger = logging.getLogger(__name__)


class GeminiImageResult(Result):
    def __init__(self, value: GeminiImageGenerateResponse, **kwargs):
        super().__init__(value, **kwargs)
        paths = []
        for candidate in value.candidates or []:
            content = candidate.content
            parts = content.parts if content else []
            for part in parts or []:
                inline_data = part.inline_data
                if inline_data is None:
                    continue
                mime_type = inline_data.mime_type or "image/png"
                data = inline_data.data
                if data is None:
                    continue
                if isinstance(data, str):
                    data = base64.b64decode(data)
                suffix = mimetypes.guess_extension(mime_type) or ".png"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                    path = tmp_file.name
                with Path(path).open("wb") as f:
                    f.write(data)
                paths.append(path)
        if not paths:
            msg = "Gemini image generation returned no images."
            raise ValueError(msg)
        self._value = paths


class GeminiImageEngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.api_key = self.config.get("DRAWING_ENGINE_API_KEY") if api_key is None else api_key
        self.model = self.config.get("DRAWING_ENGINE_MODEL") if model is None else model
        self.name = self.__class__.__name__

        if api_key is None and model is None and self.id() != "drawing":
            return

        self.transport_client = None

    def id(self) -> str:
        cfg_model = self.config.get("DRAWING_ENGINE_MODEL")
        if cfg_model and cfg_model.startswith(SUPPORTED_IMAGE_MODEL_PREFIXES):
            return "drawing"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "DRAWING_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["DRAWING_ENGINE_API_KEY"]
            # NOTE: the shared transport is stateless (headers are built per request from
            # self.api_key), so a key change needs no client rebuild.
        if "DRAWING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["DRAWING_ENGINE_MODEL"]

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)

    def forward(self, argument):
        except_remedy = argument.kwargs.get("except_remedy", None)
        request = self.build_request(argument)
        try:
            response = self.call_request(request)
        except Exception as e:
            if except_remedy is None:
                raise
            response = except_remedy(self, e, None, argument)
        return self.parse_response(response)

    def build_request(self, argument) -> EngineAPIRequest:
        prompt = argument.prop.prepared_input
        kwargs = argument.kwargs
        self.model = kwargs.get(
            "model", self.model
        )  # Important for MetadataTracker to work correctly
        operation = kwargs.get("operation")

        if operation != "create":
            msg = f"Unknown operation: {operation}"
            raise ValueError(msg)

        response_modalities = kwargs.get("response_modalities", ["IMAGE"])
        payload = GeminiImageGenerateRequest(
            contents=[GeminiImageContent(parts=[GeminiImagePart(text=prompt)])],
            generation_config=GeminiImageGenerationConfig(
                response_modalities=list(response_modalities)
            ),
        )
        return EngineAPIRequest(
            provider="google",
            operation="models.generateContent",
            payload=payload,
            method="POST",
            url=f"{GEMINI_API_BASE}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            timeout=self.client_timeout,
        )

    def call_request(self, request: EngineAPIRequest) -> GeminiImageGenerateResponse:
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        return GeminiImageGenerateResponse.model_validate(response.json())

    def parse_response(self, response: GeminiImageGenerateResponse):
        result = GeminiImageResult(response)
        metadata = {"raw_output": response}
        return [result], metadata
