import httpx
import pytest

from symai.backend.engines.neurosymbolic.llama_cpp.engine import LlamaCppEngine
from symai.backend.engines.neurosymbolic.llama_cpp.models import (
    LLAMACPP_MODEL_SPECS,
    SUPPORTED_MODELS,
    TESTED_LLAMA_CPP_COMMIT,
    LlamaCppResponse,
)
from symai.backend.settings import SYMSERVER_CONFIG
from symai.components import MetadataTracker
from tests.engines.mock_api import DUMMY_KEY, MockAPI
from tests.engines.neurosymbolic.interface import NeurosymbolicEngineTestInterface

SERVER_ENDPOINT = f"http://{SYMSERVER_CONFIG.get('--host')}:{SYMSERVER_CONFIG.get('--port')}"


class TestLlamaCppEngine(NeurosymbolicEngineTestInterface):
    engine_cls = LlamaCppEngine
    supported_models = tuple(SUPPORTED_MODELS)
    model_specs = LLAMACPP_MODEL_SPECS
    default_model = "llamacpp"
    response_cls = LlamaCppResponse
    wire_provider = "llamacpp"
    wire_operation = "chat.completions.create"
    wire_url = f"{SERVER_ENDPOINT}/v1/chat/completions"
    supports_streaming = True
    api_pinned = TESTED_LLAMA_CPP_COMMIT
    wire_model_present = False
    live_max_tokens = 512
    supports_token_counting = True

    def require_live(self, engine_api_mode):
        if engine_api_mode != "live":
            pytest.skip("use --engine-api=live to run live engine API requests")
        if not SYMSERVER_CONFIG.get("online"):
            pytest.skip("llama.cpp server is not online (start it with symserver)")
        return ""

    def assert_auth_headers(self, headers):
        content_type = headers.get("content-type") or headers.get("Content-Type")
        assert content_type == "application/json"
        auth = headers.get("authorization") or headers.get("Authorization")
        assert auth == f"Bearer {DUMMY_KEY}"

    def test_id_requires_supported_model_and_api_key(self):
        # NOTE: llama.cpp has no API key; the gate is the llama* model name.
        engine = self.engine_cls(api_key="", model="gpt-4")

        assert engine.id() != self.engine_id
        with pytest.raises(ValueError, match="not configured"):
            engine.forward(self.make_prepared_argument())

    def spec_for(self, _model="llamacpp"):
        return self.model_specs["llamacpp"]

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
        assert isinstance(raw_output, LlamaCppResponse)
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

    def test_build_request_treats_empty_stop_as_unset(self):
        engine = self.make_engine()

        empty_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ""}))
        assert "stop" not in empty_stop.body()

        user_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ["END"]}))
        assert user_stop.body()["stop"] == ["END"]

    def test_build_request_keeps_llamacpp_defaults(self):
        request = self.make_engine().build_request(self.make_prepared_argument())
        body = request.body()

        assert body["temperature"] == 0.6
        assert body["top_p"] == 0.95
        assert body["min_p"] == 0.05
        assert body["top_k"] == 40
        assert body["repeat_penalty"] == 1

    def test_compute_required_tokens_uses_server_tokenize(self):
        engine = self.make_engine(client_max_retries=0)
        rendered = "<|im_start|>user\none two<|im_end|>\n"

        def handler(request):
            if request.url.path == "/apply-template":
                return httpx.Response(200, json={"prompt": rendered}, request=request)
            return httpx.Response(200, json={"tokens": [1, 2, 3, 4, 5, 6, 7]}, request=request)

        with MockAPI(engine, handler):
            tokens = engine.compute_required_tokens([{"role": "user", "content": "one two"}])

        assert tokens == 7

    def test_server_helper_requests_send_bearer(self):
        # NOTE: --api-key deployments 401 on unauthenticated helper/discovery calls too.
        engine = self.make_engine(client_max_retries=0)

        def handler(request):
            if request.url.path == "/props":
                return httpx.Response(
                    200,
                    json={"default_generation_settings": {"n_ctx": 8192}},
                    request=request,
                )
            if request.url.path == "/apply-template":
                return httpx.Response(200, json={"prompt": "rendered"}, request=request)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1, 2, 3]}, request=request)
            if request.url.path == "/detokenize":
                return httpx.Response(200, json={"content": "abc"}, request=request)
            msg = f"unexpected request: {request.url.path}"
            raise AssertionError(msg)

        with MockAPI(engine, handler) as api:
            n_ctx = engine._server_context_tokens()
            prompt = engine._apply_template([{"role": "user", "content": "hi"}])
            tokens = engine._tokenize("hi")
            text = engine._detokenize([1, 2, 3])

        assert n_ctx == 8192
        assert prompt == "rendered"
        assert tokens == [1, 2, 3]
        assert text == "abc"
        assert {request.headers["authorization"] for request in api.requests} == {
            f"Bearer {DUMMY_KEY}"
        }

    def test_init_discovery_failure_raises_clear_error(self, monkeypatch):
        def refused(*_args, **_kwargs):
            msg = "connection refused"
            raise httpx.ConnectError(msg)

        monkeypatch.setattr(httpx, "get", refused)

        with pytest.raises(ValueError, match=r"llama\.cpp server"):
            self.engine_cls(api_key=DUMMY_KEY, model="llamacpp")
