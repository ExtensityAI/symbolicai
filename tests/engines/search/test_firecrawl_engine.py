import pytest

from symai.backend.engines.search.firecrawl import FirecrawlEngine
from symai.backend.engines.search.firecrawl.models import (
    API_PINNED,
    FIRECRAWL_API_BASE,
    FirecrawlSearchResponse,
)
from symai.backend.engines.search.utils import normalize_url
from tests.engines.search.interface import DUMMY_KEY, MOCK_QUERY, SearchEngineTestInterface

pytestmark = pytest.mark.searchengine

SCRAPE_MARKDOWN = (
    "# UEFA EURO 2024\n\n"
    "Spain won the final 2-1 against England in Berlin, with Mikel Oyarzabal "
    "scoring the decisive goal in the 86th minute."
)


class TestFirecrawlEngine(SearchEngineTestInterface):
    engine_cls = FirecrawlEngine
    response_cls = FirecrawlSearchResponse
    default_model = "firecrawl"
    wire_url = f"{FIRECRAWL_API_BASE}/search"
    auth_header_name = "Authorization"
    auth_header_prefix = "Bearer "
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.search.firecrawl.models"
    # NOTE: no firecrawl key available — every live test skips without FIRECRAWL_API_KEY
    # through require_live until one is added.
    api_key_env = "FIRECRAWL_API_KEY"
    supports_scrape = True
    scrape_wire_url = f"{FIRECRAWL_API_BASE}/scrape"

    def make_engine(self, api_key=DUMMY_KEY):
        return self.engine_cls(api_key=api_key)

    def mock_response_json(self):
        return {
            "success": True,
            "id": "fc-search-0c3f5a1b9d2e4f80",
            "creditsUsed": 2,
            "data": {
                "web": [
                    {
                        "url": "https://www.uefa.com/euro2024/history/news/0290-1e1e3dd55cf8-66a6f0c60e69-1000--spain-2-1-england/",
                        "title": "Spain 2-1 England: EURO 2024 final",
                        "description": "Spain defeated England 2-1 in the UEFA EURO 2024 "
                        "final at Berlin's Olympiastadion.",
                    },
                    {
                        # NOTE: utm_ tracker on purpose — the engine must normalize it away.
                        "url": "https://www.bbc.com/sport/football/articles/c88jl2vzvl2o?utm_medium=feed",
                        "title": "Spain beat England to win Euro 2024",
                        "description": "Mikel Oyarzabal scored an 86th-minute winner as "
                        "Spain beat England 2-1 in the Euro 2024 final.",
                    },
                ],
                "news": [],
                "images": [],
            },
            "warning": None,
        }

    def response_dropping_required(self, payload):
        del payload["success"]
        return payload

    def expected_request_body_subset(self):
        return {"query": MOCK_QUERY}

    def expected_prepared_input(self, query):
        return query

    def scrape_mock_response_json(self):
        return {
            "success": True,
            "data": {
                "markdown": SCRAPE_MARKDOWN,
                "metadata": {"title": "UEFA EURO 2024", "statusCode": 200},
            },
        }

    def test_build_request_search_options_wire_aliases(self):
        engine = self.make_engine()
        argument = self.make_argument(
            kwargs={
                "limit": 5,
                "location": "Romania",
                "sources": ["web"],
                "formats": ["markdown"],
                "only_main_content": True,
                "proxy": "auto",
            }
        )
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        assert body["query"] == MOCK_QUERY
        assert body["limit"] == 5
        assert body["location"] == "Romania"
        assert body["sources"] == ["web"]
        # scrape options serialize under their camelCase wire aliases
        assert body["scrapeOptions"] == {
            "formats": ["markdown"],
            "onlyMainContent": True,
            "proxy": "auto",
        }

    def test_scrape_route_wire_shape_and_metadata(self):
        api, output, metadata = self.forward_through_mock(
            payload=self.scrape_mock_response_json(), url=self.scrape_url()
        )

        assert api.last_body == {
            "url": normalize_url(self.scrape_url()),
            "formats": ["markdown"],
        }
        assert output[0].value == SCRAPE_MARKDOWN
        assert metadata["final_url"] == normalize_url(self.scrape_url())

    def test_max_chars_per_result_truncates_descriptions(self):
        _api, output, _metadata = self.forward_through_mock(max_chars_per_result=20)

        assert "..." in output[0].value

    @pytest.mark.engine_live
    def test_live_search_comprehensive(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(
            query="cine este nicusor dan",
            kwargs={
                "limit": 5,
                "location": "Romania",
                "sources": ["web"],
                "formats": ["markdown"],
                "only_main_content": True,
                "proxy": "auto",
            },
        )
        engine.prepare(argument)
        output, metadata = engine.forward(argument)
        res = output[0]

        assert res is not None
        assert isinstance(res._value, str)
        assert len(res._value) > 0

        raw = metadata["raw_output"]
        assert isinstance(raw, dict)
        assert "web" in raw

        assert hasattr(res, "get_citations")
        citations = res.get_citations()
        assert isinstance(citations, list)
        assert len(citations) > 0

        for citation in citations:
            assert isinstance(citation.id, int)
            assert isinstance(citation.url, str)
            assert citation.start <= citation.end

    @pytest.mark.engine_live
    def test_live_search_domain_filter(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        domains = ["arxiv.org", "nature.com"]
        filters = " OR ".join(f"site:{domain}" for domain in domains)
        argument = self.make_argument(
            query=f"({filters}) machine learning",
            kwargs={
                "limit": 10,
                "max_chars_per_result": 500,
                "formats": ["markdown"],
                "proxy": "auto",
            },
        )
        engine.prepare(argument)
        output, metadata = engine.forward(argument)
        res = output[0]

        assert isinstance(res._value, str)
        assert len(res._value) > 0

        web_results = metadata["raw_output"].get("web", [])
        assert isinstance(web_results, list)
        assert len(web_results) > 0

        citations = res.get_citations()
        assert isinstance(citations, list)
        assert len(citations) > 0
