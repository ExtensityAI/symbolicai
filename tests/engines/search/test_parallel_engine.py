import re
from urllib.parse import urlparse

import httpx
import pytest

from symai.backend.engines.search.parallel import ParallelEngine
from symai.backend.engines.search.parallel.models import (
    API_PINNED,
    PARALLEL_API_BASE,
    PARALLEL_EXTRACT_PATH,
    PARALLEL_SEARCH_PATH,
    ParallelExtractResponse,
    ParallelSearchResponse,
    ParallelUsageItem,
)
from symai.components import MetadataTracker
from tests.engines.mock_api import MockAPI
from tests.engines.search.interface import DUMMY_KEY, MOCK_QUERY, SearchEngineTestInterface

pytestmark = pytest.mark.searchengine


def assert_bulletproof_citations(res):
    """Validate citation integrity with exhaustive checks (Parallel marker layout)."""
    assert hasattr(res, "get_citations"), "Result must expose get_citations()"
    assert isinstance(res._value, str) and len(res._value) > 0

    citations = res.get_citations()
    assert isinstance(citations, list)

    all_markers = re.findall(r"\[(\d+)\]", res.value)

    if len(citations) == 0:
        assert not all_markers, "Markers found in text but citations list is empty"
        return

    assert not re.search(r"\[[^\]]+\]\(https?://[^)]+\)", res.value)
    assert "(, , )" not in res.value
    assert "()" not in res.value

    assert all_markers, "No citation markers were found in the output"

    seen_ids = set()
    seen_urls = set()
    prev_end = -1

    for c in citations:
        assert isinstance(c.id, int)
        assert c.id not in seen_ids, f"Duplicate citation ID: {c.id}"
        seen_ids.add(c.id)

        assert 0 <= c.start <= c.end <= len(res.value)
        assert c.start > prev_end, (
            f"Citation {c.id} overlaps with previous (start={c.start}, prev_end={prev_end})"
        )
        prev_end = c.end

        slice_text = res.value[c.start : c.end]
        assert slice_text == f"[{c.id}]", (
            f"Marker mismatch: expected '[{c.id}]', got '{slice_text}'"
        )

        parsed = urlparse(c.url)
        assert parsed.scheme, f"Citation {c.id} URL missing scheme: {c.url}"
        assert parsed.netloc, f"Citation {c.id} URL missing netloc: {c.url}"
        assert "utm_" not in c.url

        assert c.url not in seen_urls, f"Duplicate URL in citations: {c.url}"
        seen_urls.add(c.url)

        assert isinstance(c.title, str) and len(c.title) > 0, f"Citation {c.id} has empty title"

    assert sorted(int(m) for m in all_markers) == sorted(c.id for c in citations)
    assert sorted(c.id for c in citations) == list(range(1, len(citations) + 1))


class TestParallelEngine(SearchEngineTestInterface):
    engine_cls = ParallelEngine
    response_cls = ParallelSearchResponse
    default_model = "parallel"
    wire_url = f"{PARALLEL_API_BASE}{PARALLEL_SEARCH_PATH}"
    auth_header_name = "x-api-key"
    auth_header_prefix = ""
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.search.parallel.models"
    api_key_env = "PARALLEL_API_KEY"
    supports_scrape = True
    scrape_wire_url = f"{PARALLEL_API_BASE}{PARALLEL_EXTRACT_PATH}"

    def make_engine(self, api_key=DUMMY_KEY):
        return self.engine_cls(api_key=api_key)

    def mock_response_json(self):
        return {
            "search_id": "search_0c3f5a1b9d2e4f80",
            "session_id": "sess_0c3f5a1b9d2e4f80",
            "results": [
                {
                    "url": "https://www.uefa.com/euro2024/history/news/0290-1e1e3dd55cf8-66a6f0c60e69-1000--spain-2-1-england/",
                    "title": "Spain 2-1 England: EURO 2024 final",
                    "publish_date": "2024-07-14",
                    "excerpts": [
                        "Spain won UEFA EURO 2024 with a 2-1 victory over England in the "
                        "final in Berlin. [Oyarzabal](https://tracking.example.com/click) "
                        "scored the winning goal."
                    ],
                },
                {
                    # NOTE: utm_ tracker on purpose — the engine must normalize it away.
                    "url": "https://www.bbc.com/sport/football/articles/c88jl2vzvl2o?utm_source=feed",
                    "title": "Spain beat England to win record fourth Euro title",
                    "publish_date": "2024-07-14",
                    "excerpts": [
                        "Mikel Oyarzabal's 86th-minute strike gave Spain a 2-1 win over "
                        "England in the Euro 2024 final in Berlin."
                    ],
                },
            ],
            "warnings": None,
            # Realistic GA wire shape: a list of billed SKU items (typed as
            # ParallelUsageItem so MetadataTracker SKU accounting stays live).
            "usage": [{"name": "sku_search", "count": 1}],
        }

    def response_dropping_required(self, payload):
        del payload["session_id"]
        return payload

    def expected_request_body_subset(self):
        return {"mode": "advanced", "search_queries": [MOCK_QUERY]}

    def expected_prepared_input(self, query):
        return [query]

    def scrape_mock_response_json(self):
        return {
            "extract_id": "extract_0c3f5a1b9d2e4f80",
            "session_id": "sess_1d4g6b2c0e3f5g91",
            "results": [
                {
                    "url": self.scrape_url(),
                    "title": "UEFA EURO 2024",
                    "excerpts": [
                        "Official tournament site: Spain lifted the trophy after a 2-1 "
                        "win over England in the Berlin final."
                    ],
                    "full_content": None,
                }
            ],
            "errors": [],
            "warnings": [],
            "usage": [{"name": "sku_extract_excerpts", "count": 3}],
        }

    @pytest.mark.parametrize("mode", ["basic", "advanced", "turbo"])
    def test_build_request_forwards_mode(self, mode):
        engine = self.make_engine()
        argument = self.make_argument(kwargs={"mode": mode})
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        assert body["mode"] == mode

    def test_build_request_normalizes_source_policy_domains(self):
        engine = self.make_engine()
        argument = self.make_argument(
            kwargs={
                "allowed_domains": [
                    "tomshardware.com",
                    "https://www.arstechnica.com",  # scheme is stripped
                    "tomshardware",  # no registrable TLD -> dropped
                ],
                "excluded_domains": ["https://www.pinterest.com"],
            }
        )
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        source_policy = body["advanced_settings"]["source_policy"]
        # NOTE: tldextract's fqdn keeps the www subdomain (the utils docstring claims
        # it is dropped — reported engine/doc mismatch); pin the actual behavior.
        assert source_policy["include_domains"] == ["tomshardware.com", "www.arstechnica.com"]
        assert source_policy["exclude_domains"] == ["www.pinterest.com"]

    def test_extract_route_returns_typed_response(self):
        _api, output, metadata = self.forward_through_mock(
            payload=self.scrape_mock_response_json(), url=self.scrape_url()
        )

        assert "Spain" in output[0].value
        assert isinstance(metadata["raw_output"], ParallelExtractResponse)
        assert metadata["final_url"] == self.scrape_url()

    def test_usage_items_are_typed(self):
        # Regression: usage typed as JsonValue left items as plain dicts, silently
        # breaking MetadataTracker SKU accounting (it reads item.name / item.count).
        _api, _output, metadata = self.forward_through_mock()

        usage = metadata["raw_output"].usage
        assert all(isinstance(item, ParallelUsageItem) for item in usage)
        assert usage[0].name == "sku_search"
        assert usage[0].count == 1

    def test_usage_tracker_accumulates_sku_counts(self):
        # Regression pin: components.py MetadataTracker's ParallelEngine branch must
        # accumulate SKU counts from raw_output.usage items across calls.
        engine = self.make_engine()

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                argument = self.make_argument()
                engine.prepare(argument)
                engine.forward(argument)
                engine.forward(argument)
            usage = tracker.usage

        details = usage[("ParallelEngine", None)]
        assert details["usage"]["total_calls"] == 2
        assert details["extras"]["sku_search"] == 2

    def test_build_request_forwards_session_id_and_client_model(self):
        # Regression: the old SDK-based engine forwarded session_id / client_model;
        # both are real GA v1 wire fields and must reach the body.
        engine = self.make_engine()
        argument = self.make_argument(
            kwargs={"session_id": "sess_abc123", "client_model": "gpt-5.5"}
        )
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        assert body["session_id"] == "sess_abc123"
        assert body["client_model"] == "gpt-5.5"

    def test_extract_request_forwards_session_id_and_client_model(self):
        engine = self.make_engine()
        argument = self.make_argument(
            url=self.scrape_url(),
            kwargs={"session_id": "sess_abc123", "client_model": "gpt-5.5"},
        )
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        assert body["session_id"] == "sess_abc123"
        assert body["client_model"] == "gpt-5.5"

    def test_task_route_forwards_location_as_advanced_settings(self):
        # Regression: the old engine mapped `location` onto the task create request's
        # advanced_settings (a real GA v1 wire field: TaskAdvancedSettings.location).
        engine = self.make_engine()
        argument = self.make_argument(kwargs={"processor": "lite", "location": "ro"})
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        assert body["advanced_settings"] == {"location": "ro"}

    @pytest.mark.parametrize("alias", ["task_output_schema", "task_output", "output"])
    def test_task_route_accepts_output_schema_aliases(self, alias):
        # Regression: the old engine fell back through these aliases to output_schema.
        engine = self.make_engine()
        argument = self.make_argument(
            kwargs={"processor": "lite", alias: "Summarize in one sentence."}
        )
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        assert body["task_spec"]["output_schema"]["type"] == "text"
        assert body["task_spec"]["output_schema"]["description"] == ("Summarize in one sentence.")

    def test_task_route_accepts_timeout_aliases(self):
        # Regression: the old engine accepted bare `timeout` / `api_timeout` as
        # aliases for the task-poll knobs; the explicit task_* names win.
        engine = self.make_engine()
        argument = self.make_argument(
            kwargs={"processor": "lite", "timeout": 120, "api_timeout": 30}
        )
        engine.prepare(argument)

        options = engine.build_request(argument).call_options

        assert options.task_timeout == 120
        assert options.task_api_timeout == 30

        argument = self.make_argument(
            kwargs={"processor": "lite", "timeout": 120, "task_timeout": 60}
        )
        engine.prepare(argument)

        assert engine.build_request(argument).call_options.task_timeout == 60

    @pytest.mark.engine_live
    @pytest.mark.parametrize("mode", ["basic", "advanced", "turbo"])
    def test_live_search_citations_and_formatting(self, engine_api_mode, mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(
            query="President of Romania 2025 inauguration timeline and partner",
            kwargs={"mode": mode},
        )
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)

        assert_bulletproof_citations(output[0])

    @pytest.mark.engine_live
    def test_live_search_domain_filtering(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        domains = [
            "tomshardware.com",
            "https://www.arstechnica.com",
            "tomshardware",  # invalid, should be ignored
        ]
        argument = self.make_argument(
            query="what is the best gpu", kwargs={"allowed_domains": domains}
        )
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)
        res = output[0]

        citation_netlocs = {urlparse(c.url).netloc for c in res.get_citations()}
        # Parallel API includes apex; cite hosts may include www.
        assert any(
            n in citation_netlocs
            for n in (
                "www.tomshardware.com",
                "tomshardware.com",
                "www.arstechnica.com",
                "arstechnica.com",
            )
        ), "No citations from allowed domains found"

    @pytest.mark.engine_live
    def test_live_search_location_geo_targeting(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(query="stiri Romania actualitate", kwargs={"location": "ro"})
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)
        res = output[0]

        assert_bulletproof_citations(res)

        citations = res.get_citations()
        if len(citations) > 0:
            citation_domains = {urlparse(c.url).netloc.lower() for c in citations}
            has_ro_domain = any(domain.endswith(".ro") for domain in citation_domains)
            assert has_ro_domain, (
                f"Expected at least one .ro domain with location='ro', got: {citation_domains}"
            )

    @pytest.mark.engine_live
    @pytest.mark.parametrize("processor", ["lite-fast"])
    def test_live_task_route_via_processor(self, engine_api_mode, processor):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(
            query="Romania housing price index 2010-2025",
            kwargs={"processor": processor, "task_api_timeout": 600},
        )
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)
        res = output[0]

        assert hasattr(res, "get_citations"), "Task route must still return SearchResult interface"
        # raw is a TaskRunResult with .output and .run attributes
        assert hasattr(res.raw, "output") and hasattr(res.raw, "run")
        assert res.raw.output is not None

        citations = res.get_citations()
        assert isinstance(citations, list)
        assert res.value
