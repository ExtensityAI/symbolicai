import pytest

from symai.backend.engines.ocr.mistral import MistralOCREngine
from symai.backend.engines.ocr.mistral.models import (
    API_PINNED,
    MISTRAL_OCR_URL,
    MistralOCRResponse,
)
from tests.engines.ocr.interface import MOCK_DOCUMENT_URL, OCREngineTestInterface

pytestmark = pytest.mark.ocrengine

PAGE_ONE_MARKDOWN = "# Sample Document\n\nThis is the first page of the sample PDF."
PAGE_TWO_MARKDOWN = "A table on the second page.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
SAMPLE_IMAGE_BASE64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD"


class TestMistralOCREngine(OCREngineTestInterface):
    engine_cls = MistralOCREngine
    response_cls = MistralOCRResponse
    default_model = "mistral-ocr-latest"
    wire_url = MISTRAL_OCR_URL
    auth_header_name = "Authorization"
    auth_header_prefix = "Bearer "
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.ocr.mistral.models"
    api_key_env = "MISTRAL_API_KEY"
    # NOTE: small (~13KB) stable public PDF for the live smoke.
    live_document_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    def mock_response_json(self):
        return {
            "pages": [
                {
                    "index": 0,
                    "markdown": PAGE_ONE_MARKDOWN,
                    "images": [
                        {
                            "id": "img-0.jpeg",
                            "top_left_x": 100,
                            "top_left_y": 200,
                            "bottom_right_x": 300,
                            "bottom_right_y": 400,
                            "image_base64": SAMPLE_IMAGE_BASE64,
                        }
                    ],
                    "dimensions": {"dpi": 200, "height": 2200, "width": 1700},
                },
                {
                    "index": 1,
                    "markdown": PAGE_TWO_MARKDOWN,
                    "images": [],
                    "dimensions": {"dpi": 200, "height": 2200, "width": 1700},
                },
            ],
            "model": "mistral-ocr-latest",
            "usage_info": {"pages_processed": 2, "doc_size_bytes": 13268},
            "document_annotation": None,
        }

    def response_dropping_required(self, payload):
        del payload["pages"]
        return payload

    def expected_request_body_subset(self):
        return {
            "model": self.default_model,
            "document": {"type": "document_url", "document_url": MOCK_DOCUMENT_URL},
        }
