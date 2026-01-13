import pytest

from symai.backend.settings import SYMAI_CONFIG
from symai.extended import Interface

API_KEY = SYMAI_CONFIG.get("SEARCH_ENGINE_API_KEY")
MODEL_FIRECRAWL = str(SYMAI_CONFIG.get("SEARCH_ENGINE_MODEL", "")).lower() == "firecrawl"


pytestmark = [
    pytest.mark.searchengine,
    pytest.mark.skipif(not API_KEY, reason="SEARCH_ENGINE_API_KEY not set; live test skipped"),
    pytest.mark.skipif(not MODEL_FIRECRAWL, reason="SEARCH_ENGINE_MODEL is not 'firecrawl'."),
]


def _iface():
    try:
        return Interface("firecrawl")
    except Exception as e:
        pytest.skip(f"Firecrawl interface initialization failed: {e}")


def test_firecrawl_search_comprehensive():
    """
    Test comprehensive search with location, language, scraping, formatting, and citations.
    Combines: limit, location, language, sources, onlyMainContent, format, proxy.

    Note: More parameters are supported (tbs, scrape_location, maxAge, etc.).
    Check Firecrawl v2 API docs for full parameter list.
    """
    search = _iface()
    query = "cine este nicusor dan"
    res = search.search(
        query,
        limit=5,
        location="Romania",
        sources=["web"],
        formats=["markdown"],
        only_main_content=True,
        proxy="auto"
    )

    assert res is not None
    assert isinstance(res._value, str)
    assert len(res._value) > 0

    assert isinstance(res.raw, dict)
    assert "web" in res.raw

    assert hasattr(res, "get_citations")
    citations = res.get_citations()
    assert isinstance(citations, list)
    assert len(citations) > 0

    for citation in citations:
        assert hasattr(citation, "id")
        assert hasattr(citation, "title")
        assert hasattr(citation, "url")
        assert hasattr(citation, "start")
        assert hasattr(citation, "end")
        assert isinstance(citation.id, int)
        assert isinstance(citation.url, str)
        assert citation.start <= citation.end


def test_firecrawl_search_domain_filter():
    """
    Test domain-filtered search with site: query format and max_chars_per_result.
    Query format: "(site:arxiv.org OR site:nature.com) machine learning"
    Combines: domain filter, limit, max_chars, format, proxy.

    Note: More parameters are supported (tbs, sources, scrape_location, etc.).
    Check Firecrawl v2 API docs for full parameter list.
    """
    search = _iface()

    domains = ["arxiv.org", "nature.com"]
    filters = " OR ".join(f"site:{domain}" for domain in domains)
    base_query = "machine learning"
    query = f"({filters}) {base_query}"

    res = search.search(
        query,
        limit=10,
        max_chars_per_result=500,
        formats=["markdown"],
        proxy="auto"
    )

    assert isinstance(res._value, str)
    assert len(res._value) > 0

    assert isinstance(res.raw, dict)
    web_results = res.raw.get("web", [])
    assert isinstance(web_results, list)
    assert len(web_results) > 0

    citations = res.get_citations()
    assert isinstance(citations, list)
    assert len(citations) > 0
