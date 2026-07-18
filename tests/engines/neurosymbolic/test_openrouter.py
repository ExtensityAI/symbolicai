import base64
from pathlib import Path

import httpx

from symai.backend.engines.neurosymbolic.openrouter.engine import (
    OPENROUTER_CHAT_COMPLETIONS_URL,
    OpenRouterEngine,
)
from symai.backend.engines.neurosymbolic.openrouter.models import (
    API_PINNED,
    OPENROUTER_MODEL_SPECS,
    SUPPORTED_OPENROUTER_MODELS,
    OpenRouterResponse,
    openrouter_strip_prefix,
)
from symai.components import MetadataTracker
from tests.engines.interface import MockAPI, NeurosymbolicEngineTestInterface


class TestOpenRouterEngine(NeurosymbolicEngineTestInterface):
    engine_cls = OpenRouterEngine
    supported_models = tuple(SUPPORTED_OPENROUTER_MODELS)
    model_specs = OPENROUTER_MODEL_SPECS
    default_model = "openrouter:moonshotai/kimi-k2.5"
    response_cls = OpenRouterResponse
    wire_provider = "openrouter"
    wire_operation = "chat.completions.create"
    wire_url = OPENROUTER_CHAT_COMPLETIONS_URL
    supports_streaming = True
    api_pinned = API_PINNED

    def spec_for(self, model):
        return self.model_specs[openrouter_strip_prefix(model)]

    def expected_wire_model(self, model=None):
        return openrouter_strip_prefix(model or self.default_model)

    def vision_messages(self, image_path):
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode()
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                    {"type": "text", "text": "What is in this image? Answer in one word."},
                ],
            }
        ]

    def mock_response_json(self):
        return {
            "id": "gen-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "2",
                        "reasoning": "Add one and one.",
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "Add one and one."}
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.00002,
                "is_byok": False,
                "prompt_tokens_details": {
                    "cached_tokens": 4,
                    "cache_write_tokens": 2,
                    "audio_tokens": 0,
                    "video_tokens": 0,
                },
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
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"reasoning": "Add one "}}]},
                {"choices": [{"index": 0, "delta": {"reasoning": "and one."}}]},
                {"choices": [{"index": 0, "delta": {"content": "2"}}]},
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ]
        )

    def mock_tool_call_json(self):
        payload = self.mock_response_json()
        payload["choices"][0]["message"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                }
            ],
        }
        return payload

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
        assert isinstance(raw_output, OpenRouterResponse)
        assert raw_output.choices[0].finish_reason == "stop"
        assert raw_output.usage.total_tokens == 15

    def test_usage_tracking_includes_cache_breakdowns(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                engine.forward(self.make_prepared_argument(kwargs={"max_tokens": 16}))
            details = tracker.usage[(self.engine_cls.__name__, self.default_model)]

        assert details["prompt_breakdown"]["cached_tokens"] == 4
        assert details["extras"]["cache_write_tokens"] == 2

    def test_build_request_strips_provider_prefix_from_wire_model(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert request.body()["model"] == "moonshotai/kimi-k2.5"

    def test_build_request_remaps_max_completion_tokens_alias(self):
        request = self.make_engine().build_request(
            self.make_prepared_argument(kwargs={"max_completion_tokens": 32})
        )
        body = request.body()

        assert "max_completion_tokens" not in body
        assert body["max_tokens"] == 32

        explicit = self.make_engine().build_request(
            self.make_prepared_argument(kwargs={"max_tokens": 16, "max_completion_tokens": 32})
        )
        assert explicit.body()["max_tokens"] == 16

    def test_build_request_treats_empty_stop_as_unset(self):
        engine = self.make_engine()

        empty_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ""}))
        assert "stop" not in empty_stop.body()

        user_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ["END"]}))
        assert user_stop.body()["stop"] == ["END"]
