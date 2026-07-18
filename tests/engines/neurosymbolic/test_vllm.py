import httpx
import pytest

from symai.backend.engines.neurosymbolic.vllm.engine import VLLMEngine
from symai.backend.engines.neurosymbolic.vllm.models import (
    SUPPORTED_MODELS,
    TESTED_VLLM_COMMIT,
    VLLM_MODEL_SPECS,
    VLLMResponse,
)
from symai.backend.settings import SYMSERVER_CONFIG
from symai.components import MetadataTracker
from tests.engines.interface import MockAPI, NeurosymbolicEngineTestInterface

SERVER_ENDPOINT = f"http://{SYMSERVER_CONFIG.get('--host') or 'localhost'}:{SYMSERVER_CONFIG.get('--port') or 8001}"


class TestVLLMEngine(NeurosymbolicEngineTestInterface):
    engine_cls = VLLMEngine
    supported_models = tuple(SUPPORTED_MODELS)
    model_specs = VLLM_MODEL_SPECS
    default_model = "vllm"
    response_cls = VLLMResponse
    wire_provider = "vllm"
    wire_operation = "chat.completions.create"
    wire_url = f"{SERVER_ENDPOINT}/v1/chat/completions"
    supports_streaming = True
    api_pinned = TESTED_VLLM_COMMIT
    live_max_tokens = 512
    supports_token_counting = True

    def require_live(self, engine_api_mode):
        if engine_api_mode != "live":
            pytest.skip("use --engine-api=live to run live engine API requests")
        if not SYMSERVER_CONFIG.get("online"):
            pytest.skip("vLLM server is not online (start it with symserver)")
        return ""

    def assert_auth_headers(self, headers):
        content_type = headers.get("content-type") or headers.get("Content-Type")
        assert content_type == "application/json"

    def spec_for(self, _model="vllm"):
        return self.model_specs["vllm"]

    def expected_wire_model(self, _model=None):
        return self.make_engine().server_model

    def test_id_requires_supported_model_and_api_key(self):
        # NOTE: vLLM has no API key; the gate is the vllm* model name.
        engine = self.engine_cls(api_key="", model="gpt-4")

        assert engine.id() != self.engine_id
        with pytest.raises(ValueError, match="not configured"):
            engine.forward(self.make_prepared_argument())

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
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        }

    def response_dropping_content(self, payload):
        del payload["choices"][0]["message"]["content"]
        return payload

    def response_dropping_usage(self, payload):
        del payload["usage"]
        return payload

    def mock_tool_call_json(self):
        payload = self.mock_response_json()
        payload["choices"][0]["message"] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                    "id": "call_1",
                }
            ],
        }
        return payload

    def mock_sse_body(self):
        return self.sse_body(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"reasoning_content": "Add one "}}]},
                {"choices": [{"index": 0, "delta": {"reasoning_content": "and one."}}]},
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
        assert isinstance(raw_output, VLLMResponse)
        assert raw_output.usage.total_tokens == 15

    def test_usage_tracking_accumulates_tokens(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                engine.forward(self.make_prepared_argument(kwargs={"max_tokens": 16}))
                engine.forward(self.make_prepared_argument(kwargs={"max_tokens": 16}))
            details = tracker.usage[(self.engine_cls.__name__, self.default_model)]

        assert details["usage"]["prompt_tokens"] == 20
        assert details["usage"]["completion_tokens"] == 10
        assert details["usage"]["total_calls"] == 2

    def test_build_request_treats_empty_stop_as_unset(self):
        engine = self.make_engine()

        empty_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ""}))
        assert "stop" not in empty_stop.body()

        user_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ["END"]}))
        assert user_stop.body()["stop"] == ["END"]

    def test_compute_required_tokens_uses_server_tokenize(self):
        engine = self.make_engine(client_max_retries=0)

        def handler(request):
            n = len(request.content.decode().split())
            return httpx.Response(
                200,
                json={"count": n, "tokens": list(range(n)), "max_model_len": 4096},
                request=request,
            )

        with MockAPI(engine, handler):
            tokens = engine.compute_required_tokens([{"role": "user", "content": "one two"}])

        assert tokens == 2 + 6  # tokenized words plus template overhead estimate
