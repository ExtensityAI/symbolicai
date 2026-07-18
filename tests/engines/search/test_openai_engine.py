import re
from urllib.parse import urlparse

import pytest

from symai.backend.engines.search.openai import OpenAISearchEngine
from symai.backend.engines.search.openai.models import (
    API_PINNED,
    OPENAI_RESPONSES_URL,
    OpenAISearchResponse,
)
from tests.engines.search.interface import MOCK_QUERY, SearchEngineTestInterface

pytestmark = pytest.mark.searchengine


def _answer_text_and_annotations():
    text = (
        "Spain won the UEFA Euro 2024 final, defeating England 2-1 in Berlin. "
        "Mikel Oyarzabal scored the winning goal in the 86th minute."
    )
    annotations = [
        {
            "type": "url_citation",
            "url": "https://www.uefa.com/euro2024/history/news/0290-1e1e3dd55cf8-66a6f0c60e69-1000--spain-2-1-england/",
            "title": "Spain 2-1 England: EURO 2024 final",
            "start_index": 0,
            "end_index": text.index(" in Berlin"),
        },
        {
            "type": "url_citation",
            # NOTE: utm_ tracker on purpose — the engine must normalize it away.
            "url": "https://www.bbc.com/sport/football/articles/c88jl2vzvl2o?utm_source=feed",
            "title": "Spain 2-1 England: Oyarzabal wins Euro 2024 final",
            "start_index": text.index("Mikel"),
            "end_index": len(text),
        },
    ]
    return text, annotations


class TestOpenAISearchEngine(SearchEngineTestInterface):
    engine_cls = OpenAISearchEngine
    response_cls = OpenAISearchResponse
    default_model = "gpt-4.1"
    wire_url = OPENAI_RESPONSES_URL
    auth_header_name = "Authorization"
    auth_header_prefix = "Bearer "
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.search.openai.models"
    api_key_env = "OPENAI_API_KEY"

    def mock_response_json(self):
        text, annotations = _answer_text_and_annotations()
        return {
            "id": "resp_0c3f5a1b9d2e4f80",
            "object": "response",
            "created_at": 1752800000,
            "status": "completed",
            "model": "gpt-4.1",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_0c3f5a1b9d2e4f80",
                    "status": "completed",
                    "action": {"type": "search", "query": "UEFA Euro 2024 final winner score"},
                },
                {
                    "type": "message",
                    "id": "msg_0c3f5a1b9d2e4f80",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": annotations}],
                },
            ],
            "usage": {
                "input_tokens": 318,
                "output_tokens": 64,
                "total_tokens": 382,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    def response_dropping_required(self, payload):
        # usage is optional, but when present its token fields are required.
        payload["usage"] = {"input_tokens": 318}
        return payload

    def expected_request_body_subset(self):
        return {
            "model": self.default_model,
            "input": [{"role": "system"}, {"role": "user", "content": MOCK_QUERY}],
            "tools": [{"type": "web_search"}],
            # non-reasoning default model forces the web_search tool
            "tool_choice": {"type": "web_search"},
        }

    def test_build_request_normalizes_allowed_domains(self):
        engine = self.make_engine()
        argument = self.make_argument(
            kwargs={
                "allowed_domains": [
                    "tomshardware.com",
                    "https://www.arstechnica.com",  # scheme is stripped
                    "tomshardware",  # no registrable TLD -> dropped
                ]
            }
        )
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        # NOTE: tldextract's fqdn keeps the www subdomain (the utils docstring claims
        # it is dropped — reported engine/doc mismatch); pin the actual behavior.
        assert body["tools"][0]["filters"]["allowed_domains"] == [
            "tomshardware.com",
            "www.arstechnica.com",
        ]

    @pytest.mark.engine_live
    @pytest.mark.parametrize("model", ["gpt-4.1-mini", "o3"])
    def test_live_citations_and_formatting(self, engine_api_mode, model):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(
            query="President of Romania 2025 inauguration timeline and partner (with citations)",
            kwargs={"model": model},
        )
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)
        res = output[0]

        # 1) No leftover markdown link patterns or empty parentheses artifacts
        assert not re.search(r"\[[^\]]+\]\(https?://[^)]+\)", res.value)
        assert "(, , )" not in res.value
        assert "()" not in res.value

        # 2) Citations exist with integer ids and normalized URLs (no utm_ params)
        citations = res.get_citations()
        assert isinstance(citations, list) and len(citations) >= 1
        seen_ids = set()
        for c in citations:
            assert isinstance(c.id, int)
            assert c.id not in seen_ids
            seen_ids.add(c.id)
            assert "utm_" not in c.url

            # Slice should match the marker format "[id] (title)\n"
            slice_text = res.value[c.start : c.end]
            assert slice_text.startswith(f"[{c.id}] (")
            assert slice_text.endswith(")\n")
            assert slice_text == f"[{c.id}] ({c.title})\n"

        # 3) Formatting: at least one marker pattern with newline is present
        assert re.search(r"\[\d+\] \([^)]+\)\n", res.value)

    @pytest.mark.engine_live
    @pytest.mark.parametrize("model", ["gpt-4.1", "o3"])
    def test_live_domain_filtering(self, engine_api_mode, model):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        domains = [
            "tomshardware.com",  # ok
            "https://www.arstechnica.com",  # ok, the engine yields the root domain
            "tomshardware",  # not ok
        ]
        argument = self.make_argument(
            query="what is the best gpu",
            kwargs={"model": model, "allowed_domains": domains},
        )
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)
        res = output[0]

        allowed_netlocs = {"www.tomshardware.com", "www.arstechnica.com"}
        citation_netlocs = {urlparse(c.url).netloc for c in res.get_citations()}
        assert allowed_netlocs & citation_netlocs, "No citations from allowed domains found"
