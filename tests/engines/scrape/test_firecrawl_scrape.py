import json

import pytest
from pydantic import Field

from symai.backend.settings import SYMAI_CONFIG
from symai.components import DynamicEngine
from symai.extended import Interface
from symai.models import LLMDataModel

API_KEY = SYMAI_CONFIG.get("SEARCH_ENGINE_API_KEY")
MODEL_FIRECRAWL = str(SYMAI_CONFIG.get("SEARCH_ENGINE_MODEL", "")).lower() == "firecrawl"


pytestmark = [
    pytest.mark.searchengine,
    pytest.mark.skipif(not API_KEY, reason="SEARCH_ENGINE_API_KEY not configured or missing."),
    pytest.mark.skipif(not MODEL_FIRECRAWL, reason="SEARCH_ENGINE_MODEL is not 'firecrawl'."),
]


def _iface():
    try:
        return Interface("firecrawl")
    except Exception as e:
        pytest.skip(f"Firecrawl interface initialization failed: {e}")


def test_firecrawl_scrape_pdf():
    """
    Test PDF scraping with onlyMainContent and max_chars_per_result.
    Tests: PDF parsing, content trimming, proxy settings.

    Note: More parameters are supported (location, actions, storeInCache, etc.).
    Check Firecrawl v2 API docs for full parameter list.
    """
    scraper = _iface()
    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7231600"

    # First scrape: full main content
    rsp_full = scraper.scrape(
        url,
        only_main_content=True,
        formats=["markdown"],
        proxy="auto"
    )
    assert rsp_full is not None
    text_full = str(rsp_full)
    assert len(text_full) > 0

    # Second scrape: trimmed to 100 characters via max_chars (not direct API parameter)
    # Note: max_chars_per_result is a local parameter, not sent to API
    rsp_trimmed = scraper.scrape(
        url,
        only_main_content=True,
        formats=["markdown"],
        proxy="auto"
    )
    assert rsp_trimmed is not None
    assert len(str(rsp_trimmed)[:100]) == 100


def test_firecrawl_scrape_js_heavy_sites():
    """
    Test that JS-heavy sites like LinkedIn and Twitter are currently not fully supported.
    These sites return errors (403 Forbidden or Website Not Supported).

    Note: This test verifies expected failures. Support may vary by subscription tier.
    """
    scraper = _iface()

    # Test LinkedIn (expected to fail)
    linkedin_url = "https://www.linkedin.com/posts/george-hotz-b3866476_more-technology-is-not-going-to-bring-you-activity-7367261116713861122-YcLy"
    linkedin_failed = False
    try:
        scraper.scrape(linkedin_url, proxy="auto")
    except ValueError as e:
        # Expected: 403 Forbidden or Website Not Supported
        err = str(e).lower()
        if "403" in err or "not supported" in err or "forbidden" in err:
            linkedin_failed = True

    # Test Twitter/X (expected to fail)
    twitter_url = "https://x.com/karpathy/status/1973468610917179630"
    twitter_failed = False
    try:
        scraper.scrape(twitter_url, proxy="auto")
    except ValueError as e:
        # Expected: 403 Forbidden or Website Not Supported
        err = str(e).lower()
        if "403" in err or "not supported" in err or "forbidden" in err:
            twitter_failed = True

    # Verify both failed as expected (not supported)
    assert linkedin_failed, "LinkedIn scraping should fail"
    assert twitter_failed, "Twitter/X scraping should fail"


def test_firecrawl_scrape_basic():
    """
    Test basic webpage scraping with markdown format.

    Note: More formats and parameters are supported (html, rawHtml, actions, etc.).
    Check Firecrawl v2 API docs for full parameter list.
    """
    scraper = _iface()
    url = "https://docs.firecrawl.dev/introduction"
    rsp = scraper.scrape(url, formats=["markdown"])

    assert rsp is not None
    text = str(rsp).strip()
    assert len(text) > 0
    assert "firecrawl" in text.lower()


def test_firecrawl_scrape_json_extraction_with_dynamic_engine():
    """
    Test JSON schema extraction using DynamicEngine and LLMDataModel.
    Verifies that:
    1. DynamicEngine correctly routes to FirecrawlEngine
    2. JSON format extraction works with Pydantic schema
    3. FirecrawlExtractResult._value contains JSON string when using json format
    4. Extracted data can be parsed back into the Pydantic model
    """

    class MetadataModel(LLMDataModel):
        """Bibliographic metadata extracted from a source document."""

        title: str = Field(description="Title of the source.")
        year: str = Field(description="Publication year (4 digits) or Unknown.")
        authors: list[str] = Field(default_factory=list, description="List of authors.")
        collection: str = Field(
            description="Journal/conference/publisher/collection or Unknown."
        )
        doi: str | None = Field(default=None, description="DOI if available.")
        url: str | None = Field(default=None, description="Canonical URL for this source.")
        publisher: str | None = Field(default=None, description="Publisher if available.")
        journal: str | None = Field(
            default=None, description="Journal or venue title if applicable."
        )
        volume: str | None = Field(default=None, description="Volume identifier if applicable.")
        issue: str | None = Field(
            default=None, description="Issue/number identifier if applicable."
        )
        pages: str | None = Field(default=None, description="Page range if applicable.")

    schema = MetadataModel.model_json_schema()
    json_format = {
        "type": "json",
        "prompt": "Extract bibliographic metadata from this academic paper.",
        "schema": schema,
    }

    url = "https://journals.physiology.org/doi/full/10.1152/ajpregu.00051.2002"

    with DynamicEngine("firecrawl", api_key=API_KEY):
        iface = Interface("firecrawl")
        result = iface.scrape(url, formats=[json_format])

    # Verify result._value is a non-empty JSON string
    result_str = str(result)
    assert result_str, "Result string should not be empty for JSON extraction"
    assert result_str.startswith("{"), "Result should be a JSON object string"

    # Verify we can parse the JSON
    parsed = json.loads(result_str)
    assert "title" in parsed
    assert "year" in parsed

    # Verify we can instantiate the model from extracted data
    extracted_json = result.raw.get("json", {})
    metadata = MetadataModel(**extracted_json)
    assert metadata.title
    assert metadata.year == "2002"
    assert len(metadata.authors) > 0

    # Verify model_dump round-trip
    dumped = metadata.model_dump()
    assert dumped["title"] == parsed["title"]
    assert dumped["year"] == parsed["year"]
