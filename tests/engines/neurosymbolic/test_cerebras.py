import httpx
import pytest

from symai.backend.engines.neurosymbolic.cerebras.engine import (
    CEREBRAS_CHAT_COMPLETIONS_URL,
    CerebrasEngine,
)
from symai.backend.engines.neurosymbolic.cerebras.models import (
    API_PINNED,
    CEREBRAS_MODEL_SPECS,
    SUPPORTED_CEREBRAS_MODELS,
    CerebrasResponse,
    cerebras_strip_prefix,
)
from symai.components import MetadataTracker
from tests.engines.interface import MockAPI, NeurosymbolicEngineTestInterface


class TestCerebrasEngine(NeurosymbolicEngineTestInterface):
    engine_cls = CerebrasEngine
    supported_models = tuple(SUPPORTED_CEREBRAS_MODELS)
    model_specs = CEREBRAS_MODEL_SPECS
    default_model = "cerebras:gpt-oss-120b"
    response_cls = CerebrasResponse
    wire_provider = "cerebras"
    wire_operation = "chat.completions.create"
    wire_url = CEREBRAS_CHAT_COMPLETIONS_URL
    supports_streaming = True
    api_pinned = API_PINNED

    def spec_for(self, model):
        return self.model_specs[cerebras_strip_prefix(model)]

    def expected_wire_model(self, model=None):
        return cerebras_strip_prefix(model or self.default_model)

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
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                    "reasoning_tokens": 3,
                },
            },
            "time_info": {"queue_time": 0.001, "total_time": 0.01},
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
        assert isinstance(raw_output, CerebrasResponse)
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
        assert details["extras"]["prompt_cached_tokens"] == 4

    def test_build_request_strips_provider_prefix_from_wire_model(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert request.body()["model"] == "gpt-oss-120b"

    def test_build_request_rejects_unsupported_reasoning_effort(self):
        engine = self.make_engine()

        with pytest.raises(ValueError, match="Unsupported reasoning_effort"):
            engine.build_request(self.make_prepared_argument(kwargs={"reasoning_effort": "none"}))

    def test_build_request_treats_empty_stop_as_unset(self):
        engine = self.make_engine()

        empty_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ""}))
        assert "stop" not in empty_stop.body()

        user_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ["END"]}))
        assert user_stop.body()["stop"] == ["END"]

    def test_build_request_normalizes_flat_json_schema_response_format(self):
        engine = self.make_engine()
        flat = {"type": "json_schema", "name": "answer", "schema": {"type": "object"}}

        request = engine.build_request(
            self.make_prepared_argument(kwargs={"response_format": flat})
        )

        assert request.body()["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True},
        }

    def test_build_request_forces_n_to_one(self):
        request = self.make_engine().build_request(self.make_prepared_argument(kwargs={"n": 3}))

        assert request.body()["n"] == 1
