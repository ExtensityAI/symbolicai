from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import httpx

from symai.backend.base import Engine
from symai.backend.engines.ocr.mistral.models import (
    MISTRAL_FILES_URL,
    MISTRAL_OCR_URL,
    MistralDocumentURLChunk,
    MistralFileSchema,
    MistralImageURLChunk,
    MistralOCRRequest,
    MistralOCRResponse,
    MistralSignedURLResponse,
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

silence_noisy_loggers()

logger = logging.getLogger(__name__)


class MistralOCRResult(Result):
    """Result wrapper for Mistral OCR API responses."""

    def __init__(self, value: MistralOCRResponse, per_page: bool = False, **kwargs):
        raw = value.model_dump()
        super().__init__(raw, **kwargs)
        pages = raw["pages"]
        if per_page:
            self._value = [page["markdown"] for page in pages]
        else:
            self._value = "\n\n".join(page["markdown"] for page in pages)
        # build image mapping: id -> base64 data URI (only populated when include_image_base64=True)
        self._images = {}
        for page in pages:
            for img in page["images"]:
                b64 = img.get("image_base64")
                if b64:
                    self._images[img["id"]] = b64

    @property
    def images(self) -> dict[str, str]:
        """Mapping of image id to base64 data URI. Empty when include_image_base64 was not set."""
        return self._images

    def __str__(self) -> str:
        if isinstance(self._value, list):
            return "\n\n---\n\n".join(self._value)
        return self._value or ""


class MistralOCREngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = api_key or self.config.get("OCR_ENGINE_API_KEY")
        self.model = model or self.config.get("OCR_ENGINE_MODEL", "mistral-ocr-latest")
        self.name = self.__class__.__name__
        self.transport_client = None
        # NOTE: per_page steers result parsing, not the wire payload; stash it for
        # parse_response, which only receives the response.
        self._per_page = False

        if self.id() == super().id():
            return

        if not self.api_key:
            msg = "Mistral API key not found. Set OCR_ENGINE_API_KEY in config or environment."
            raise ValueError(msg)

    def id(self) -> str:
        if self.config.get("OCR_ENGINE_API_KEY") and self.config.get(
            "OCR_ENGINE_MODEL", ""
        ).lower().startswith("mistral"):
            return "ocr"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "OCR_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["OCR_ENGINE_API_KEY"]
        if "OCR_ENGINE_MODEL" in kwargs:
            self.model = kwargs["OCR_ENGINE_MODEL"]

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def build_request(self, argument) -> EngineAPIRequest:
        kwargs = argument.kwargs
        self._per_page = kwargs.get("per_page", False)

        document_url = getattr(argument.prop, "document_url", None)
        image_url = getattr(argument.prop, "image_url", None)

        assert document_url or image_url, "Provide document_url or image_url."

        if document_url:
            resolved = self._resolve_local_file(document_url)
            document = MistralDocumentURLChunk(document_url=resolved)
        else:
            resolved = self._resolve_local_file(image_url)
            document = MistralImageURLChunk(image_url=resolved)

        ocr_kwargs: dict = {"model": self.model, "document": document}

        # pass through Mistral-specific options from kwargs
        for key in (
            "table_format",
            "extract_header",
            "extract_footer",
            "include_image_base64",
            "pages",
            "image_limit",
            "image_min_size",
        ):
            if key in kwargs:
                ocr_kwargs[key] = kwargs[key]

        payload = MistralOCRRequest.model_validate(ocr_kwargs)

        return EngineAPIRequest(
            provider="mistral",
            operation="ocr",
            payload=payload,
            method="POST",
            url=MISTRAL_OCR_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.client_timeout,
        )

    def call_request(self, request: EngineAPIRequest) -> MistralOCRResponse:
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        return MistralOCRResponse.model_validate(response.json())

    def parse_response(self, response: MistralOCRResponse):
        return [MistralOCRResult(response, per_page=self._per_page)], {"raw_output": response}

    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "MistralOCREngine does not support processed_input."
        )
        document_url = getattr(argument.prop, "document_url", None)
        image_url = getattr(argument.prop, "image_url", None)
        assert document_url or image_url, "MistralOCREngine requires 'document_url' or 'image_url'."
        argument.prop.prepared_input = document_url or image_url

    def _http_client(self) -> httpx.Client:
        return (
            self.transport_client
            if self.transport_client is not None
            else default_engine_api_client()
        )

    def _resolve_local_file(self, url):
        """If url is a local file, upload to Mistral and return a signed HTTPS URL."""
        # already a remote URL or inline data — nothing to resolve
        if url.startswith(("http://", "https://", "data:")):
            return url
        path = Path(url.removeprefix("file://"))
        if not path.is_file():
            return url
        file_id = self._upload_file(path)
        return self._signed_url(file_id)

    def _upload_file(self, path: Path) -> str:
        """POST /v1/files (multipart, purpose=ocr) and return the uploaded file id."""
        try:
            response = self._http_client().post(
                MISTRAL_FILES_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (path.name, path.read_bytes())},
                data={"purpose": "ocr"},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            msg = f"Mistral OCR file upload failed: {e}"
            raise RuntimeError(msg) from e
        return MistralFileSchema.model_validate(response.json()).id

    def _signed_url(self, file_id: str) -> str:
        """GET /v1/files/{file_id}/url and return the signed HTTPS URL."""
        try:
            response = self._http_client().get(
                f"{MISTRAL_FILES_URL}/{file_id}/url",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"expiry": 1},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            msg = f"Mistral OCR signed-url request failed: {e}"
            raise RuntimeError(msg) from e
        return MistralSignedURLResponse.model_validate(response.json()).url
