import pytest

from symai.backend.settings import SYMAI_CONFIG
from symai.components import MetadataTracker
from symai.extended import Interface
from symai.utils import RuntimeInfo

API_KEY = SYMAI_CONFIG.get("OCR_ENGINE_API_KEY")
MODEL_MISTRAL = SYMAI_CONFIG.get("OCR_ENGINE_MODEL", "").lower().startswith("mistral")

pytestmark = [
    pytest.mark.ocrengine,
    pytest.mark.skipif(not API_KEY, reason="OCR_ENGINE_API_KEY not set; live test skipped"),
    pytest.mark.skipif(not MODEL_MISTRAL, reason="OCR_ENGINE_MODEL does not start with 'mistral'."),
]

ARXIV_PDF_URL = "https://arxiv.org/pdf/2507.11768"


def _iface():
    try:
        return Interface("ocr")
    except Exception as e:
        pytest.skip(f"OCR interface initialization failed: {e}")


def test_mistral_ocr_full_markdown():
    """OCR a PDF and return assembled markdown (default)."""
    ocr = _iface()
    res = ocr(document_url=ARXIV_PDF_URL)

    assert res is not None
    assert isinstance(res._value, str)
    assert len(res._value) > 0


def test_mistral_ocr_per_page():
    """OCR a PDF with per_page=True returns list[str]."""
    ocr = _iface()
    res = ocr(document_url=ARXIV_PDF_URL, include_image_base64=True, per_page=True)

    assert res is not None
    assert isinstance(res._value, list)
    assert len(res._value) > 0
    assert all(isinstance(page, str) for page in res._value)


def test_mistral_ocr_metadata_tracker():
    """MetadataTracker captures page-based usage from Mistral OCR."""
    ocr = _iface()
    with MetadataTracker() as tracker:
        res = ocr(document_url=ARXIV_PDF_URL)

    assert res is not None

    # verify raw metadata was captured
    assert len(tracker.metadata) > 0

    # verify usage accumulation
    usage = tracker.usage
    assert len(usage) > 0

    for (engine_name, _model_name), data in usage.items():
        assert engine_name == "MistralOCREngine"
        assert data["usage"]["total_calls"] >= 1
        assert "extras" in data
        assert data["extras"]["pages_processed"] > 0

    # verify RuntimeInfo round-trip
    def _ocr_pricing(info, pricing):
        return info.extras.get("pages_processed", 0) * pricing.get("per_page", 0)

    usage_per_engine = RuntimeInfo.from_tracker(tracker, 0)
    total = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)
    for _, data in usage_per_engine.items():
        total += RuntimeInfo.estimate_cost(data, _ocr_pricing, pricing={"per_page": 0.002})

    assert total.total_calls >= 1
    assert total.extras.get("pages_processed", 0) > 0
    assert total.cost_estimate > 0
