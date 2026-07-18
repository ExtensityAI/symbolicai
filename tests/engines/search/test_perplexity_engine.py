import pytest

from symai.backend.engines.search.perplexity import PerplexityEngine
from symai.backend.engines.search.perplexity.models import (
    API_PINNED,
    PERPLEXITY_CHAT_COMPLETIONS_URL,
    PerplexityResponse,
)
from tests.engines.search.interface import MOCK_QUERY, SearchEngineTestInterface

pytestmark = pytest.mark.searchengine


class TestPerplexityEngine(SearchEngineTestInterface):
    engine_cls = PerplexityEngine
    response_cls = PerplexityResponse
    default_model = "sonar"
    wire_url = PERPLEXITY_CHAT_COMPLETIONS_URL
    auth_header_name = "Authorization"
    auth_header_prefix = "Bearer "
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.search.perplexity.models"
    api_key_env = "PERPLEXITY_API_KEY"

    def mock_response_json(self):
        return {
            "id": "chatcmpl-0c3f5a1b9d2e4f80",
            "object": "chat.completion",
            "created": 1752800000,
            "model": "sonar",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Spain won the UEFA Euro 2024 final, beating England 2-1[1]. "
                            "Mikel Oyarzabal scored the decisive goal in the 86th minute[2]."
                        ),
                    },
                }
            ],
            "citations": [
                "https://www.uefa.com/euro2024/history/news/0290-1e1e3dd55cf8-66a6f0c60e69-1000--spain-2-1-england/",
                # NOTE: utm_ tracker on purpose — the engine must normalize it away.
                "https://www.bbc.com/sport/football/articles/c88jl2vzvl2o?utm_campaign=feed",
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 41, "total_tokens": 53},
        }

    def response_dropping_required(self, payload):
        del payload["choices"]
        return payload

    def expected_request_body_subset(self):
        return {
            "model": self.default_model,
            "messages": [{"role": "system"}, {"role": "user", "content": MOCK_QUERY}],
        }

    def test_build_request_applies_default_sampling_kwargs(self):
        engine = self.make_engine()
        argument = self.make_argument()
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.9
        assert body["top_k"] == 0
        assert body["presence_penalty"] == 0
        assert body["frequency_penalty"] == 1
        assert body["search_recency_filter"] == "month"
        assert body["return_images"] is False
        assert body["return_related_questions"] is False
        assert body["search_domain_filter"] == []

    def test_build_request_forwards_provider_kwargs(self):
        engine = self.make_engine()
        argument = self.make_argument(
            kwargs={
                "max_tokens": 64,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 10,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "search_domain_filter": ["arxiv.org", "nature.com"],
                "return_images": True,
                "return_related_questions": True,
                "search_recency_filter": "week",
                "web_search_options": {"search_context_size": "high"},
            }
        )
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.8
        assert body["top_k"] == 10
        assert body["presence_penalty"] == 0.5
        assert body["frequency_penalty"] == 0.3
        assert body["search_domain_filter"] == ["arxiv.org", "nature.com"]
        assert body["return_images"] is True
        assert body["return_related_questions"] is True
        assert body["search_recency_filter"] == "week"
        assert body["web_search_options"] == {"search_context_size": "high"}

    def test_prepare_honors_custom_system_message(self):
        engine = self.make_engine()
        argument = self.make_argument(kwargs={"system_message": "Explain like I'm five years old."})

        engine.prepare(argument)

        assert argument.prop.prepared_input[0] == {
            "role": "system",
            "content": "Explain like I'm five years old.",
        }

    @pytest.mark.engine_live
    def test_live_basic_query(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(query="What is the capital of France?")
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)
        res = output[0]

        assert res._value is not None
        assert isinstance(res._value, str) and len(res._value) > 0
        assert "Paris" in res._value, "Response for capital of France should contain 'Paris'."
