from __future__ import annotations

import base64
import logging
import mimetypes
import tempfile
from pathlib import Path

from symai.backend.base import Engine
from symai.backend.engines.drawing.openai.models import (
    OPENAI_IMAGES_EDITS_URL,
    OPENAI_IMAGES_GENERATIONS_URL,
    OPENAI_IMAGES_VARIATIONS_URL,
    OpenAIImageEditsRequest,
    OpenAIImageGenerationsRequest,
    OpenAIImagesResponse,
    OpenAIImageVariationsRequest,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import (
    DEFAULT_RETRIES,
    default_engine_api_client,
    execute_engine_api_request,
)
from symai.symbol import Result
from symai.utils import silence_noisy_loggers

# silence noisy libraries
silence_noisy_loggers("openai")

logger = logging.getLogger(__name__)

# Fallback timeout (seconds) for image URL downloads when the engine has no
# explicit client_timeout configured (dall-e URLs expire after 60 minutes).
DOWNLOAD_TIMEOUT_SECONDS = 60.0


class GPTImageResult(Result):
    """
    Wraps an OpenAI Images API response (generations / edits / variations).
    Exposes .value as the raw response and ._value as the list of local
    image file paths (URLs downloaded, b64_json decoded to temp files).
    """

    def __init__(self, value, image_paths, **kwargs):
        super().__init__(value, **kwargs)
        self._value = image_paths


class GPTImageEngine(Engine):
    """
    A drop-in engine for OpenAI's unified Images API,
    supporting gpt-image-1, dall-e-2, dall-e-3,
    with all the extra parameters (background, moderation, etc).
    Raw REST via the shared httpx transport — no openai SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        super().__init__()
        self.config = SYMAI_CONFIG
        # pick up a separate config slot if you like, or fall back
        self.api_key = self.config.get("DRAWING_ENGINE_API_KEY") if api_key is None else api_key
        self.model = self.config.get("DRAWING_ENGINE_MODEL") if model is None else model
        self.name = self.__class__.__name__
        self.transport_client = None

    def id(self) -> str:
        # register this engine under "gpt-image" by default
        cfg_model = self.config.get("DRAWING_ENGINE_MODEL")
        if cfg_model.startswith("gpt-image-") or cfg_model.startswith("dall-e-"):
            return "drawing"
        return super().id()

    def command(self, *args, **kwargs):
        """
        Allow hot-swapping API key or model at runtime.
        """
        super().command(*args, **kwargs)
        if "DRAWING_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["DRAWING_ENGINE_API_KEY"]
        if "DRAWING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["DRAWING_ENGINE_MODEL"]

    def prepare(self, argument):
        """
        Simply copy processed_input → prepared_input
        """
        argument.prop.prepared_input = str(argument.prop.processed_input)

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def build_request(self, argument) -> EngineAPIRequest:
        prompt = argument.prop.prepared_input
        kwargs = dict(argument.kwargs)

        model = kwargs.get("model", self.model)
        operation = kwargs.get("operation")

        if operation is None:
            msg = "Operation not specified!"
            raise ValueError(msg)

        n = kwargs.get("n", 1)

        self._normalize_size(kwargs)

        if operation == "create":
            return self._build_generations_request(prompt, model, n, kwargs)
        if operation == "variation":
            return self._build_variations_request(model, n, kwargs)
        if operation == "edit":
            return self._build_edits_request(prompt, model, n, kwargs)
        msg = f"Unknown image operation: {operation}"
        raise ValueError(msg)

    def call_request(self, request: EngineAPIRequest) -> OpenAIImagesResponse:
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        return OpenAIImagesResponse.model_validate(response.json())

    def parse_response(self, response: OpenAIImagesResponse):
        paths = []
        for item in response.data:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                path = tmp_file.name
            if item.url is not None:
                self._download_image(item.url, path)
            elif item.b64_json is not None:
                Path(path).write_bytes(base64.b64decode(item.b64_json))
            paths.append(path)
        return [GPTImageResult(response, paths)], {"raw_output": response}

    def _download_image(self, url: str, path: str) -> None:
        # NOTE: the download rides the engine's transport client (mockable in tests);
        # dall-e URLs expire after 60 minutes, so an unbounded timeout is never right.
        client = self.transport_client or default_engine_api_client()
        timeout = (
            self.client_timeout if self.client_timeout is not None else DOWNLOAD_TIMEOUT_SECONDS
        )
        response = client.get(url, follow_redirects=True, timeout=timeout)
        response.raise_for_status()
        Path(path).write_bytes(response.content)

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # NOTE: multipart requests must NOT set Content-Type — httpx adds it with the boundary.
        if content_type is not None:
            headers["Content-Type"] = content_type
        return headers

    def _build_generations_request(self, prompt, model, n, kwargs) -> EngineAPIRequest:
        payload_kwargs = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": kwargs.get("size"),
        }

        if model == "dall-e-3":
            payload_kwargs["response_format"] = kwargs.get("response_format", "url")
            payload_kwargs["quality"] = kwargs.get("quality", "standard")
            payload_kwargs["style"] = kwargs.get("style", "vivid")

        if model.startswith("gpt-image-"):
            payload_kwargs["quality"] = kwargs.get("quality", "medium")
            payload_kwargs["moderation"] = kwargs.get("moderation", "auto")
            payload_kwargs["background"] = kwargs.get("background", "auto")
            # NOTE: the legacy engine read `output_compression` into output_format by
            # mistake; the wire field is output_format ("png" | "jpeg" | "webp").
            payload_kwargs["output_format"] = kwargs.get("output_format", "png")
            if payload_kwargs["output_format"] in ("jpeg", "webp"):
                payload_kwargs["output_compression"] = kwargs.get("output_compression", 100)

        return EngineAPIRequest(
            provider="openai",
            operation="images/generations",
            payload=OpenAIImageGenerationsRequest.model_validate(payload_kwargs),
            method="POST",
            url=OPENAI_IMAGES_GENERATIONS_URL,
            headers=self._headers("application/json"),
            timeout=self.client_timeout,
        )

    def _build_variations_request(self, model, n, kwargs) -> EngineAPIRequest:
        assert "image_path" in kwargs, "image_path required for variation"
        payload = OpenAIImageVariationsRequest.model_validate(
            {
                "model": model,
                "n": n,
                "size": kwargs.get("size"),
                "response_format": kwargs.get("response_format", "url"),
            }
        )
        files = [("image", self._file_part(kwargs["image_path"]))]
        return EngineAPIRequest(
            provider="openai",
            operation="images/variations",
            payload=payload,
            files=files,
            method="POST",
            url=OPENAI_IMAGES_VARIATIONS_URL,
            headers=self._headers(),
            timeout=self.client_timeout,
        )

    def _build_edits_request(self, prompt, model, n, kwargs) -> EngineAPIRequest:
        assert "image_path" in kwargs, "image_path required for edit"
        img_paths = kwargs["image_path"]
        if not isinstance(img_paths, (list, tuple)):
            img_paths = [img_paths]
        # NOTE: the wire field mirrors the SDK's brackets array format (openai-python
        # extract_files, array_format="brackets"): one source image rides as `image`,
        # multiple source images as repeated `image[]` parts. Bytes are read eagerly
        # so transport retries never hit an exhausted file handle.
        image_field = "image" if len(img_paths) == 1 else "image[]"
        files = [(image_field, self._file_part(p)) for p in img_paths]

        mask_path = kwargs.get("mask_path")
        if mask_path is not None:
            files.append(("mask", self._file_part(mask_path)))

        payload_kwargs = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": kwargs.get("size"),
        }
        if model.startswith("gpt-image-"):
            payload_kwargs["quality"] = kwargs.get("quality", "auto")

        return EngineAPIRequest(
            provider="openai",
            operation="images/edits",
            payload=OpenAIImageEditsRequest.model_validate(payload_kwargs),
            files=files,
            method="POST",
            url=OPENAI_IMAGES_EDITS_URL,
            headers=self._headers(),
            timeout=self.client_timeout,
        )

    @staticmethod
    def _file_part(path) -> tuple[str, bytes, str]:
        path = Path(path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return (path.name, path.read_bytes(), mime)

    def _normalize_size(self, kwargs):
        if "size" in kwargs and isinstance(kwargs["size"], int):
            s = kwargs["size"]
            kwargs["size"] = f"{s}x{s}"
