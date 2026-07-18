import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines.neurosymbolic.deepseek.engine import (
    DEEPSEEK_CHAT_COMPLETIONS_URL,
    DeepSeekXReasoningEngine,
)
from symai.backend.engines.neurosymbolic.deepseek.models import (
    API_PINNED,
    DEEPSEEK_MODEL_SPECS,
    SUPPORTED_MODELS,
    DeepSeekResponse,
    deepseek_strip_prefix,
)
from symai.components import MetadataTracker
from tests.engines.interface import MockAPI, NeurosymbolicEngineTestInterface


class TestDeepSeekEngine(NeurosymbolicEngineTestInterface):
    engine_cls = DeepSeekXReasoningEngine
    supported_models = tuple(SUPPORTED_MODELS)
    model_specs = DEEPSEEK_MODEL_SPECS
    default_model = "deepseek:deepseek-v4-flash"
    response_cls = DeepSeekResponse
    wire_provider = "deepseek"
    wire_operation = "chat.completions.create"
    wire_url = DEEPSEEK_CHAT_COMPLETIONS_URL
    supports_streaming = True
    api_pinned = API_PINNED

    def spec_for(self, model):
        return self.model_specs[deepseek_strip_prefix(model)]

    def expected_wire_model(self, model=None):
        return deepseek_strip_prefix(model or self.default_model)

    def mock_response_json(self):
        return {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "2",
                        "reasoning_content": "Add one and one.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 3},
                "prompt_cache_hit_tokens": 4,
                "prompt_cache_miss_tokens": 6,
            },
        }

    def response_dropping_content(self, payload):
        del payload["choices"][0]["message"]["content"]
        return payload

    def response_dropping_usage(self, payload):
        del payload["usage"]
        return payload

    def mock_sse_body(self):
        return self.sse_body(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "reasoning_content": "Add one "},
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {"reasoning_content": "and one."}}]},
                {"choices": [{"index": 0, "delta": {"content": "2"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ]
        )

    def test_forward_streams_sse_and_aggregates_response(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(
                200,
                content=self.mock_sse_body(),
                headers={"content-type": "text/event-stream"},
                request=request,
            ),
        ):
            output, metadata = engine.forward(self.make_prepared_argument(kwargs={"stream": True}))

        assert output == ["2"]
        assert metadata["thinking"] == "Add one and one."
        raw_output = metadata["raw_output"]
        assert isinstance(raw_output, DeepSeekResponse)
        assert raw_output.choices[0].finish_reason == "stop"
        assert raw_output.usage.prompt_tokens == 10
        assert raw_output.usage.completion_tokens == 5
        assert raw_output.usage.total_tokens == 15

    def test_usage_tracking_includes_reasoning_and_cache_breakdowns(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                engine.forward(self.make_prepared_argument(kwargs={"max_tokens": 16}))
            details = tracker.usage[(self.engine_cls.__name__, self.default_model)]

        assert details["completion_breakdown"]["reasoning_tokens"] == 3
        assert details["extras"]["prompt_cache_hit_tokens"] == 4
        assert details["extras"]["prompt_cache_miss_tokens"] == 6

    def test_build_request_roundtrips_null_content_tool_call_messages(self):
        engine = self.make_engine()
        messages = [
            {"role": "user", "content": "What is 1+1?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        ]

        request = engine.build_request(self.make_prepared_argument(messages=messages))
        body_messages = request.body()["messages"]

        assert "content" not in body_messages[1]
        assert body_messages[1]["tool_calls"] == [{"id": "call_1", "type": "function"}]
        assert body_messages[2]["content"] == "2"

        with pytest.raises(ValidationError):
            engine.build_request(
                self.make_prepared_argument(messages=[{"role": "user", "content": 1}])
            )

    def test_build_request_strips_provider_prefix_from_wire_model(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert request.body()["model"] == "deepseek-v4-flash"

    def test_build_request_treats_empty_stop_as_unset(self):
        engine = self.make_engine()

        empty_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ""}))
        assert empty_stop.body()["stop"] == "<|endoftext|>"

        user_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ["END"]}))
        assert user_stop.body()["stop"] == ["END"]
