import pytest

from symai.backend.settings import SYMAI_CONFIG
from symai.extended import Interface


API_KEY = bool(SYMAI_CONFIG.get("SEARCH_ENGINE_API_KEY"))
MODEL_PARALLEL = str(SYMAI_CONFIG.get("SEARCH_ENGINE_MODEL", "")).lower() == "parallel"


pytestmark = [
    pytest.mark.searchengine,
    pytest.mark.skipif(not API_KEY, reason="SEARCH_ENGINE_API_KEY not configured or missing."),
    pytest.mark.skipif(not MODEL_PARALLEL, reason="SEARCH_ENGINE_MODEL is not 'parallel'."),
]


def _iface():
    try:
        return Interface("parallel")
    except Exception as e:
        pytest.skip(f"Parallel interface initialization failed: {e}")


def test_parallel_scrape_basic():
    scraper = _iface()
    url = "https://trafilatura.readthedocs.io/en/latest/crawls.html"
    rsp = scraper.scrape(url)
    assert rsp is not None
    text = str(rsp).strip()
    assert len(text) > 0


def test_parallel_scrape_pdf():
    scraper = _iface()
    url = "https://jevinwest.org/papers/Kim2017asa.pdf"
    rsp = scraper.scrape(url)
    assert rsp is not None
    assert len(str(rsp)) > 0


@pytest.mark.parametrize(
    "url",
    [
        "https://x.com/karpathy/status/1973468610917179630",
        "https://www.linkedin.com/posts/george-hotz-b3866476_more-technology-is-not-going-to-bring-you-activity-7367261116713861122-YcLy",
    ],
)
def test_parallel_scrape_js_heavy(url):
    scraper = _iface()
    rsp = scraper.scrape(url)
    assert rsp is not None
    assert len(str(rsp)) > 0
