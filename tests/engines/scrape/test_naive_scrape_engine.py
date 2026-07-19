"""Naive scrape engine tests: mock validates the fetch/extract pipeline offline,
live hits stable sandbox sites (--engine-api=live). No provider key required.
"""

import httpx
import pytest

from symai.backend.engines.scrape.requests import RequestsEngine, RequestsResult
from symai.core import Argument

ARTICLE_HTML = b"""
<html><head><title>Test Page</title></head>
<body><article><h1>Symbolic AI</h1>
<p>Neurosymbolic programming combines learning and reasoning in one framework.</p>
</article></body></html>
"""

REFRESH_HTML = b"""
<html><head><meta http-equiv="refresh" content="0;url=/final"></head><body></body></html>
"""


def make_argument(url: str, **kwargs) -> Argument:
    return Argument((), {}, {"url": url, **kwargs})


def mock_engine(handler, **kwargs) -> RequestsEngine:
    kwargs.setdefault("retries", 1)
    kwargs.setdefault("backoff_factor", 0)
    engine = RequestsEngine(**kwargs)
    engine.client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return engine


def run_scrape(engine: RequestsEngine, url: str, **kwargs):
    argument = make_argument(url, **kwargs)
    engine.prepare(argument)
    results, metadata = engine.forward(argument)
    return results[0], metadata


@pytest.fixture
def engine_api_mode(request):
    return request.config.getoption("--engine-api")


class TestNaiveScrapeMock:
    def test_extracts_markdown(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=ARTICLE_HTML, headers={"Content-Type": "text/html"})

        result, metadata = run_scrape(mock_engine(handler), "https://example.com/page")
        assert isinstance(result, RequestsResult)
        assert "Neurosymbolic programming" in str(result)
        assert metadata["response_source"] == "requests"

    def test_strips_utm_params(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, content=ARTICLE_HTML, headers={"Content-Type": "text/html"})

        run_scrape(
            mock_engine(handler),
            "https://example.com/page?utm_source=newsletter&utm_medium=email&keep=1",
        )
        assert seen == ["https://example.com/page?keep=1"]

    def test_seeds_bypass_cookies(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=ARTICLE_HTML, headers={"Content-Type": "text/html"})

        engine = mock_engine(handler)
        run_scrape(engine, "https://example.com/page")
        assert engine.client.cookies.get("cookieconsent_status", domain="example.com") == "allow"

    def test_follows_meta_refresh(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/final":
                return httpx.Response(
                    200, content=ARTICLE_HTML, headers={"Content-Type": "text/html"}
                )
            return httpx.Response(200, content=REFRESH_HTML, headers={"Content-Type": "text/html"})

        result, metadata = run_scrape(mock_engine(handler), "https://example.com/start")
        assert "Neurosymbolic programming" in str(result)
        assert metadata["final_url"] == "https://example.com/final"

    def test_follows_meta_refresh_legacy_charset(self):
        # No charset in Content-Type and legacy single-byte (cp1252) content *inside
        # the refresh URL*: apparent-encoding detection must decode the page well
        # enough to find the right target (httpx alone defaults to utf-8-with-
        # replacement and would mangle the URL).
        refresh_html = """
        <html><head><meta http-equiv="refresh" content="0;url=/caf\xe9"></head>
        <body><h1>Cr\xe9dit agr\xe9\xe9</h1>
        <p>Le caf\xe9 \xe0 la fa\xe7on de No\xebl, na\xefve r\xe9sum\xe9, gar\xe7on, fran\xe7ais, tr\xe8s bien, d\xe9j\xe0 vu.</p>
        </body></html>
        """.encode("cp1252")

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path != "/start":
                return httpx.Response(
                    200, content=ARTICLE_HTML, headers={"Content-Type": "text/html"}
                )
            return httpx.Response(200, content=refresh_html, headers={"Content-Type": "text/html"})

        result, metadata = run_scrape(mock_engine(handler), "https://example.com/start")
        assert "Neurosymbolic programming" in str(result)
        # legacy byte 0xE9 decodes to '\xe9'; httpx percent-encodes it as UTF-8 on the wire.
        assert metadata["final_url"] == "https://example.com/caf%C3%A9"

    def test_retries_then_succeeds(self):
        calls = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, content=ARTICLE_HTML, headers={"Content-Type": "text/html"})

        result, _ = run_scrape(mock_engine(handler, retries=2), "https://example.com/flaky")
        assert len(calls) == 2
        assert "Neurosymbolic programming" in str(result)

    def test_raises_after_retry_exhaustion(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        engine = mock_engine(handler, retries=2)
        argument = make_argument("https://example.com/down")
        engine.prepare(argument)
        with pytest.raises(httpx.HTTPStatusError):
            engine.forward(argument)


class TestNaiveScrapeLive:
    @pytest.fixture(autouse=True)
    def require_live(self, engine_api_mode):
        if engine_api_mode != "live":
            pytest.skip("live scrape test skipped in mock mode")

    def test_scrape_example_com(self):
        engine = RequestsEngine(timeout=15)
        result, metadata = run_scrape(engine, "https://example.com/")
        assert isinstance(result, RequestsResult)
        # trafilatura extracts main text; the <h1> title is not part of it
        assert "documentation examples" in str(result)
        assert metadata["final_url"].startswith("https://example.com")

    @pytest.mark.parametrize("output_format", ["txt", "markdown", "html"])
    def test_output_formats(self, output_format):
        engine = RequestsEngine(timeout=15)
        result, _ = run_scrape(engine, "https://example.com/", output_format=output_format)
        content = str(result).strip()
        assert len(content) > 0
        if output_format == "html":
            assert content.startswith("<")

    def test_pdf_extraction(self):
        engine = RequestsEngine(timeout=20)
        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        result, _ = run_scrape(engine, url)
        assert len(str(result)) > 0

    def test_render_js(self):
        pytest.importorskip("playwright.sync_api", reason="Playwright runtime required")
        engine = RequestsEngine(timeout=30)
        # quotes.toscrape.com is a scraping sandbox serving quotes only via JS
        try:
            result, metadata = run_scrape(
                engine, "https://quotes.toscrape.com/js/", render_js=True, render_timeout=30
            )
        except Exception as exc:
            if "Executable doesn't exist" in str(exc):
                pytest.skip("Playwright browser binaries not installed (playwright install)")
            raise
        assert isinstance(result, RequestsResult)
        assert metadata["response_source"] == "playwright"
        assert len(str(result)) > 0
