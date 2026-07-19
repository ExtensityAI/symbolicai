"""Mistral OCR wire models.

Locked against the Mistral OCR API (POST https://api.mistral.ai/v1/ocr) and the
Files API used to resolve local files (POST /v1/files, GET /v1/files/{id}/url).
Field names cross-checked against the mistralai 2.7.0 SDK wire models
(OCRRequest, OCRResponse, OCRPageObject, OCRUsageInfo, FileSchema).
"""

from __future__ import annotations

from typing import Literal

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

MISTRAL_API_BASE = "https://api.mistral.ai/v1"
MISTRAL_OCR_URL = f"{MISTRAL_API_BASE}/ocr"
MISTRAL_FILES_URL = f"{MISTRAL_API_BASE}/files"


class MistralDocumentURLChunk(EngineRequestPayload):
    type: Literal["document_url"] = "document_url"
    document_url: str
    document_name: str | None = None


class MistralImageURLChunk(EngineRequestPayload):
    type: Literal["image_url"] = "image_url"
    image_url: str


MistralDocument = MistralDocumentURLChunk | MistralImageURLChunk


class MistralOCRRequest(EngineRequestPayload):
    model: str
    document: MistralDocument
    # NOTE: the wire also accepts a comma/range string ("0,2-4"); we only ever
    # send the list form, so the model stays narrow on purpose.
    pages: list[int] | None = None
    include_image_base64: bool | None = None
    image_limit: int | None = None
    image_min_size: int | None = None
    table_format: Literal["markdown", "html"] | None = None
    extract_header: bool | None = None
    extract_footer: bool | None = None


class MistralOCRImage(EngineResponsePayload):
    id: str
    top_left_x: int | None = None
    top_left_y: int | None = None
    bottom_right_x: int | None = None
    bottom_right_y: int | None = None
    image_base64: str | None = None
    image_annotation: str | None = None


class MistralOCRPageDimensions(EngineResponsePayload):
    dpi: int
    height: int
    width: int


class MistralOCRPage(EngineResponsePayload):
    index: int
    markdown: str
    images: list[MistralOCRImage]
    dimensions: MistralOCRPageDimensions | None = None
    header: str | None = None
    footer: str | None = None


class MistralOCRUsageInfo(EngineResponsePayload):
    pages_processed: int
    doc_size_bytes: int | None = None


class MistralOCRResponse(EngineResponsePayload):
    pages: list[MistralOCRPage]
    model: str
    usage_info: MistralOCRUsageInfo
    document_annotation: str | None = None


class MistralFileSchema(EngineResponsePayload):
    """POST /v1/files response (only the fields the engine consumes are modeled)."""

    id: str
    object: str | None = None
    size_bytes: int | None = None
    created_at: int | None = None
    filename: str | None = None
    purpose: str | None = None


class MistralSignedURLResponse(EngineResponsePayload):
    """GET /v1/files/{file_id}/url response."""

    url: str
