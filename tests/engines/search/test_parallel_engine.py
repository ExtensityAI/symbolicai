import re
from urllib.parse import urlparse

import pytest

from symai.backend.settings import SYMAI_CONFIG
from symai.extended import Interface

API_KEY = SYMAI_CONFIG.get("SEARCH_ENGINE_API_KEY")
MODEL_PARALLEL = str(SYMAI_CONFIG.get("SEARCH_ENGINE_MODEL", "")).lower() == "parallel"


pytestmark = [
    pytest.mark.searchengine,
    pytest.mark.skipif(not API_KEY, reason="PARALLEL/SEARCH_ENGINE_API_KEY not set; live test skipped"),
    pytest.mark.skipif(not MODEL_PARALLEL, reason="SEARCH_ENGINE_MODEL is not 'parallel'."),
]


def _iface():
    try:
        return Interface("parallel")
    except Exception as e:
        pytest.skip(f"Parallel interface initialization failed: {e}")


def test_parallel_search_citations_and_formatting():
    search = _iface()
    query = "President of Romania 2025 inauguration timeline and partner"
    res = search.search(query, mode="agentic")

    assert hasattr(res, "get_citations"), "Result must expose get_citations()"
    assert isinstance(res._value, str) and len(res._value) > 0

    # No leftover markdown link patterns or empty parentheses artifacts
    assert not re.search(r"\[[^\]]+\]\(https?://[^)]+\)", res.value)
    assert "(, , )" not in res.value
    assert "()" not in res.value

    # No square brackets outside citation markers
    all_markers = re.findall(r"\[(\d+)\]", res.value)
    assert all_markers, "No citation markers were found in the output"

    citations = res.get_citations()
    assert isinstance(citations, list) and len(citations) >= 1

    seen = set()
    for c in citations:
        assert isinstance(c.id, int)
        assert c.id not in seen
        seen.add(c.id)
        assert "utm_" not in c.url
        assert 0 <= c.start <= c.end <= len(res.value)
        slice_text = res.value[c.start : c.end]
        assert slice_text == f"[{c.id}]"

    assert sorted(int(m) for m in all_markers) == sorted(c.id for c in citations)


def test_parallel_search_domain_filtering():
    search = _iface()

    domains = [
        "tomshardware.com",
        "https://www.arstechnica.com",
        "tomshardware",  # invalid, should be ignored
    ]

    query = "what is the best gpu"
    res = search.search(query, allowed_domains=domains)

    citation_netlocs = {urlparse(c.url).netloc for c in res.get_citations()}
    # Parallel API includes apex; cite hosts may include www.
    assert any(
        n in citation_netlocs
        for n in ("www.tomshardware.com", "tomshardware.com", "www.arstechnica.com", "arstechnica.com")
    ), "No citations from allowed domains found"

@pytest.mark.parametrize("processor", ["lite-fast"])
def test_parallel_task_route_via_processor(processor):
    search = _iface()

    query = "Romania housing price index 2010-2025"
    res = search.search(query, processor=processor, task_api_timeout=600)

    assert hasattr(res, "get_citations"), "Task route must still return SearchResult interface"
    assert isinstance(res.raw, dict) and "results" in res.raw
    assert res.raw.get("task_output") is not None

    citations = res.get_citations()
    assert isinstance(citations, list)
    assert res.value
