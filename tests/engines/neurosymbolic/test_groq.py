import httpx

from symai.backend.engines.neurosymbolic.groq.engine import (
    GROQ_CHAT_COMPLETIONS_URL,
    GroqEngine,
)
from symai.backend.engines.neurosymbolic.groq.models import (
    API_PINNED,
    GROQ_MODEL_SPECS,
    SUPPORTED_GROQ_MODELS,
    GroqResponse,
    groq_strip_prefix,
)
from symai.components import MetadataTracker
from tests.engines.mock_api import MockAPI
from tests.engines.neurosymbolic.interface import NeurosymbolicEngineTestInterface


class TestGroqEngine(NeurosymbolicEngineTestInterface):
    engine_cls = GroqEngine
    supported_models = tuple(SUPPORTED_GROQ_MODELS)
    model_specs = GROQ_MODEL_SPECS
    default_model = "groq:openai/gpt-oss-120b"
    response_cls = GroqResponse
    wire_provider = "groq"
    wire_operation = "chat.completions.create"
    wire_url = GROQ_CHAT_COMPLETIONS_URL
    supports_streaming = True
    api_pinned = API_PINNED
    max_tokens_wire_key = "max_completion_tokens"

    def spec_for(self, model):
        return self.model_specs[groq_strip_prefix(model)]

    def expected_wire_model(self, model=None):
        return groq_strip_prefix(model or self.default_model)

    def mock_response_json(self):
        return {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "2",
                        "reasoning": "Add one and one.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "queue_time": 0.001,
                "prompt_time": 0.002,
                "completion_time": 0.003,
                "total_time": 0.005,
                "completion_tokens_details": {"reasoning_tokens": 3},
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
        assert isinstance(raw_output, GroqResponse)
        assert raw_output.choices[0].finish_reason == "stop"
        assert raw_output.usage.total_tokens == 15

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

    def test_usage_tracking_includes_reasoning_breakdown(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                engine.forward(self.make_prepared_argument(kwargs={"max_tokens": 16}))
            details = tracker.usage[(self.engine_cls.__name__, self.default_model)]

        assert details["completion_breakdown"]["reasoning_tokens"] == 3

    def test_build_request_strips_provider_prefix_from_wire_model(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert request.body()["model"] == "openai/gpt-oss-120b"

    def test_build_request_remaps_max_tokens_alias(self):
        request = self.make_engine().build_request(
            self.make_prepared_argument(kwargs={"max_tokens": 32})
        )
        body = request.body()

        assert "max_tokens" not in body
        assert body["max_completion_tokens"] == 32

        explicit = self.make_engine().build_request(
            self.make_prepared_argument(kwargs={"max_tokens": 32, "max_completion_tokens": 64})
        )
        assert explicit.body()["max_completion_tokens"] == 64

    def test_build_request_drops_unsupported_kwargs_with_warning(self):
        request = self.make_engine().build_request(
            self.make_prepared_argument(kwargs={"logprobs": True, "search_settings": {}, "seed": 7})
        )
        body = request.body()

        assert "logprobs" not in body
        assert "search_settings" not in body
        assert body["seed"] == 7

    def test_build_request_defaults_reasoning_effort_from_spec(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert request.body()["reasoning_effort"] == "low"

    def test_build_request_treats_empty_stop_as_unset(self):
        engine = self.make_engine()

        empty_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ""}))
        assert "stop" not in empty_stop.body()

        user_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ["END"]}))
        assert user_stop.body()["stop"] == ["END"]

    def test_build_request_json_object_mode_drops_tools(self):
        engine = self.make_engine()
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

        request = engine.build_request(
            self.make_prepared_argument(
                kwargs={
                    "response_format": {"type": "json_object"},
                    "tools": tools,
                }
            )
        )
        body = request.body()

        assert "tools" not in body
        assert body["tool_choice"] == "auto"

    def test_parse_response_extracts_think_tags(self):
        engine = self.make_engine(client_max_retries=0)
        payload = self.mock_response_json()
        payload["choices"][0]["message"] = {
            "role": "assistant",
            "content": "<think>Add one and one.</think>The answer is 2.",
        }

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=payload, request=request),
        ):
            output, metadata = engine.forward(self.make_prepared_argument())

        assert output == ["The answer is 2."]
        assert metadata["thinking"] == "Add one and one."
