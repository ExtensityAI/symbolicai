"""Shared test interface for migrated engines (see ENGINE_REFACTOR_RECIPE.md).

One subclass per provider folder supplies provider facts and wire fixtures; the
interface asserts the uniform engine contract:

- registration (id gate, ENGINE_MAPPING)
- build_request wire shape (headers, params, timeout, body merge, kwarg validation)
- typed responses (fail-fast on missing content/usage)
- usage metadata tracking with pricing (MetadataTracker + spec pricing)
- streaming aggregation (skipped when the provider does not stream)
- vision (skipped when the model spec declares no vision)
- live smoke (intentionally short and cheap)

Live runs require `--engine-api=live` and the provider key in symai.config.json.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines.neurosymbolic.prompts import prompt_registry
from symai.backend.settings import SYMAI_CONFIG
from symai.components import MetadataTracker
from symai.core import Argument
from symai.functional import EngineRepository

DUMMY_KEY = "sk-test-not-a-real-key"
LIVE_PROMPT = "Reply with exactly: ok"
LIVE_TIMEOUT = 30.0


class MockAPI:
    """Routes an engine's transport through httpx.MockTransport and records requests."""

    def __init__(self, engine, handler):
        def spy(request):
            self.requests.append(request)
            return handler(request)

        self.requests = []
        self.client = httpx.Client(transport=httpx.MockTransport(spy))
        engine.transport_client = self.client

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.client.close()

    @property
    def last_request(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content.decode("utf-8"))


class EngineTestInterface:
    """Uniform contract checks for one provider engine. Subclass per provider folder."""

    # --- provider facts (override) ---
    engine_cls: ClassVar = None
    engine_id: ClassVar[str] = ""
    api_key_config: ClassVar[str] = ""
    supported_models: ClassVar[tuple] = ()
    model_specs: ClassVar[dict] = {}
    default_model: ClassVar[str] = ""
    response_cls: ClassVar = None
    wire_provider: ClassVar[str] = ""
    wire_operation: ClassVar[str] = ""
    wire_url: ClassVar[str] = ""
    supports_streaming: ClassVar[bool] = False
    api_pinned: ClassVar[str] = ""

    # --- provider hooks (override) ---
    def mock_response_json(self) -> dict:
        raise NotImplementedError

    def response_dropping_content(self, payload: dict) -> dict:
        raise NotImplementedError

    def response_dropping_usage(self, payload: dict) -> dict:
        raise NotImplementedError

    def mock_sse_body(self) -> bytes:
        raise NotImplementedError

    def vision_messages(self, _image_path: str) -> list | None:
        return None

    # --- shared helpers ---
    def make_engine(self, model=None, **kwargs):
        return self.engine_cls(api_key=DUMMY_KEY, model=model or self.default_model, **kwargs)

    def make_prepared_argument(self, kwargs=None, messages=None):
        if kwargs is None:
            kwargs = {}
        if messages is None:
            messages = [{"role": "user", "content": "hello"}]
        return SimpleNamespace(kwargs=kwargs, prop=SimpleNamespace(prepared_input=messages))

    def make_query_argument(self, text=LIVE_PROMPT, **signature_kwargs):
        argument = Argument(
            args=(text,),
            signature_kwargs=signature_kwargs,
            decorator_kwargs={"prompt": "Follow the user instruction exactly.", "examples": []},
        )
        argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
        argument.prop.processed_input = text
        return argument

    def require_live(self, engine_api_mode) -> str:
        if engine_api_mode != "live":
            pytest.skip("use --engine-api=live to run live engine API requests")
        api_key = SYMAI_CONFIG.get(self.api_key_config, "")
        if not api_key:
            pytest.skip(f"symai.config.json {self.api_key_config} is required")
        return api_key

    def make_live_engine(self, model, api_key):
        return self.engine_cls(
            api_key=api_key,
            model=model,
            client_timeout=LIVE_TIMEOUT,
            client_max_retries=0,
        )

    def expected_cost_usd(self, model, usage: dict) -> float:
        pricing = self.model_specs[model].pricing
        if pricing.cached_input is not None:
            input_cost = (
                usage.get("prompt_cache_hit_tokens", 0) * pricing.cached_input
                + usage.get("prompt_cache_miss_tokens", usage["prompt_tokens"]) * pricing.input
            )
        else:
            input_cost = usage["prompt_tokens"] * pricing.input
        return (input_cost + usage["completion_tokens"] * pricing.output) / 1_000_000

    @staticmethod
    def sse_body(chunks: list[dict]) -> bytes:
        lines = []
        for chunk in chunks:
            lines.append(f"data: {json.dumps(chunk)}")
            lines.append("")
        lines.append("data: [DONE]")
        lines.append("")
        return "\n".join(lines).encode("utf-8")

    # --- contract checks ---
    def test_api_pinned_lock_present(self):
        assert self.api_pinned, "provider models.py must declare API_PINNED"

    def test_id_requires_supported_model_and_api_key(self):
        engine = self.engine_cls(api_key="", model=self.default_model)

        assert engine.id() != self.engine_id
        with pytest.raises(ValueError, match="not configured"):
            engine.forward(self.make_prepared_argument())

    def test_model_specs_declare_capabilities_and_pricing(self):
        assert set(self.supported_models) == set(self.model_specs)
        for spec in self.model_specs.values():
            assert isinstance(spec.reasoning, bool)
            assert isinstance(spec.vision, bool)
            assert spec.context_tokens > 0
            assert spec.response_tokens > 0
            assert spec.pricing.input >= 0
            assert spec.pricing.output >= 0
            if spec.pricing.cached_input is not None:
                assert spec.pricing.cached_input <= spec.pricing.input

    def test_build_request_wire_shape(self):
        engine = self.make_engine(client_timeout=7.0)
        argument = self.make_prepared_argument(
            kwargs={
                "temperature": 0.2,
                "max_tokens": 32,
                "extra_headers": {"X-Test": "1"},
                "extra_query": {"debug": "1"},
                "extra_body": {"temperature": 9, "vendor_flag": True},
            }
        )

        request = engine.build_request(argument)
        body = request.body()

        assert request.provider == self.wire_provider
        assert request.operation == self.wire_operation
        assert request.method == "POST"
        assert request.url == self.wire_url
        assert request.headers["Authorization"] == f"Bearer {DUMMY_KEY}"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["X-Test"] == "1"
        assert request.params == {"debug": "1"}
        assert request.timeout == 7.0

        assert body["model"] == self.default_model
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 32
        assert body["vendor_flag"] is True
        assert "extra_body" not in body
        assert "extra_headers" not in body
        assert "extra_query" not in body

    def test_build_request_omits_max_tokens_when_not_provided(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert "max_tokens" not in request.body()

    def test_build_request_timeout_prefers_kwarg_then_client_timeout(self):
        engine = self.make_engine(client_timeout=7.0)
        bare_engine = self.make_engine()

        default_request = engine.build_request(self.make_prepared_argument())
        explicit_request = engine.build_request(
            self.make_prepared_argument(kwargs={"timeout": 11.0})
        )
        bare_request = bare_engine.build_request(self.make_prepared_argument())

        assert default_request.timeout == 7.0
        assert explicit_request.timeout == 11.0
        assert bare_request.timeout is None

    def test_build_request_rejects_bad_provider_kwargs(self):
        engine = self.make_engine()

        with pytest.raises(ValidationError):
            engine.build_request(self.make_prepared_argument(kwargs={"max_tokens": "32"}))

        with pytest.raises(ValueError, match="Unsupported request kwargs"):
            engine.build_request(
                self.make_prepared_argument(
                    kwargs={"strict_request_kwargs": True, "temprature": 0.2}
                )
            )

    def test_forward_mock_transport_returns_typed_response(self):
        engine = self.make_engine(client_max_retries=0)
        argument = self.make_query_argument(
            "What is 1+1?", max_tokens=16, extra_query={"debug": "1"}
        )

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ) as api:
            engine.prepare(argument)
            output, metadata = engine.forward(argument)

        assert api.last_request.method == "POST"
        assert str(api.last_request.url) == f"{self.wire_url}?debug=1"
        assert api.last_request.headers["authorization"] == f"Bearer {DUMMY_KEY}"
        assert api.last_body["model"] == self.default_model
        assert api.last_body["max_tokens"] == 16
        assert api.last_body["messages"] == argument.prop.prepared_input
        assert isinstance(output[0], str)
        assert "thinking" in metadata
        assert isinstance(metadata["raw_output"], self.response_cls)

    def test_call_request_fails_fast_when_response_drops_content(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(
                200, json=self.response_dropping_content(self.mock_response_json()), request=request
            ),
        ):
            request = engine.build_request(self.make_prepared_argument(kwargs={"max_tokens": 16}))
            with pytest.raises(ValidationError):
                engine.call_request(request)

    def test_call_request_fails_fast_when_response_drops_usage(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(
                200, json=self.response_dropping_usage(self.mock_response_json()), request=request
            ),
        ):
            request = engine.build_request(self.make_prepared_argument(kwargs={"max_tokens": 16}))
            with pytest.raises(ValidationError):
                engine.call_request(request)

    def test_usage_tracking_and_pricing(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                engine.forward(self.make_prepared_argument(kwargs={"max_tokens": 16}))
                engine.forward(self.make_prepared_argument(kwargs={"max_tokens": 16}))
            usage = tracker.usage

        details = usage[(self.engine_cls.__name__, self.default_model)]
        mock_usage = self.mock_response_json()["usage"]
        assert details["usage"]["prompt_tokens"] == 2 * mock_usage["prompt_tokens"]
        assert details["usage"]["completion_tokens"] == 2 * mock_usage["completion_tokens"]
        assert details["usage"]["total_tokens"] == 2 * mock_usage["total_tokens"]
        assert details["usage"]["total_calls"] == 2
        assert "thinking_content" not in details

        summed = {
            "prompt_tokens": details["usage"]["prompt_tokens"],
            "completion_tokens": details["usage"]["completion_tokens"],
            **details.get("extras", {}),
        }
        cost = self.expected_cost_usd(self.default_model, summed)
        assert cost > 0
        # linear pricing: cost of two identical calls is exactly twice the single-call cost
        single = self.expected_cost_usd(self.default_model, mock_usage)
        assert cost == pytest.approx(2 * single)

    def test_forward_streams_sse_and_aggregates_response(self):
        if not self.supports_streaming:
            pytest.skip(f"{self.engine_cls.__name__} does not stream")
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(
                200,
                content=self.mock_sse_body(),
                headers={"content-type": "text/event-stream"},
                request=request,
            ),
        ) as api:
            output, metadata = engine.forward(self.make_prepared_argument(kwargs={"stream": True}))

        assert api.last_body["stream"] is True
        assert api.last_body["stream_options"] == {"include_usage": True}
        assert isinstance(output[0], str) and output[0]
        raw_output = metadata["raw_output"]
        assert isinstance(raw_output, self.response_cls)
        assert raw_output.usage.total_tokens > 0

    def test_stream_fails_fast_when_usage_chunk_is_missing(self):
        if not self.supports_streaming:
            pytest.skip(f"{self.engine_cls.__name__} does not stream")
        engine = self.make_engine(client_max_retries=0)
        chunks = [{"choices": [{"index": 0, "delta": {"content": "2"}, "finish_reason": "stop"}]}]

        with MockAPI(
            engine,
            lambda request: httpx.Response(
                200,
                content=self.sse_body(chunks),
                headers={"content-type": "text/event-stream"},
                request=request,
            ),
        ):
            argument = self.make_prepared_argument(
                kwargs={"stream": True, "stream_options": {"include_usage": False}}
            )
            with pytest.raises(ValidationError):
                engine.forward(argument)

    @pytest.mark.engine_live
    def test_live_smoke(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        for model in self.supported_models:
            engine = self.make_live_engine(model, api_key)
            # NOTE: reasoning models spend budget on thinking first; 128 tokens is the
            # cheap floor that still guarantees visible content on trivial prompts.
            argument = self.make_query_argument(LIVE_PROMPT, max_tokens=128)

            engine.prepare(argument)
            output, metadata = engine.forward(argument)

            assert isinstance(output[0], str) and output[0]
            raw_output = metadata["raw_output"]
            assert isinstance(raw_output, self.response_cls)
            assert raw_output.usage.total_tokens > 0
            cost = self.expected_cost_usd(model, raw_output.usage.model_dump())
            assert 0 < cost < 0.01

    @pytest.mark.engine_live
    def test_live_stream_smoke(self, engine_api_mode):
        if not self.supports_streaming:
            pytest.skip(f"{self.engine_cls.__name__} does not stream")
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(self.default_model, api_key)
        argument = self.make_query_argument(LIVE_PROMPT, max_tokens=16, stream=True)

        engine.prepare(argument)
        output, metadata = engine.forward(argument)

        assert isinstance(output[0], str)
        assert metadata["raw_output"].usage.total_tokens > 0

    @pytest.mark.engine_live
    def test_live_vision(self, engine_api_mode):
        spec = self.model_specs[self.default_model]
        if not spec.vision:
            pytest.skip(f"{self.default_model} has no vision support")
        messages = self.vision_messages("tests/data/sample.jpg")
        if messages is None:
            pytest.skip(f"vision hook not implemented for {self.engine_cls.__name__}")
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(self.default_model, api_key)
        output, _metadata = engine.forward(self.make_prepared_argument(messages=messages))

        assert isinstance(output[0], str) and output[0]


class NeurosymbolicEngineTestInterface(EngineTestInterface):
    """Adds prompt-rendering and self-prompt checks for neurosymbolic chat engines."""

    engine_id = "neurosymbolic"
    api_key_config = "NEUROSYMBOLIC_ENGINE_API_KEY"

    def test_prepare_renders_neurosymbolic_prompt_template(self):
        engine = self.make_engine()
        argument = Argument(
            args=("What changed?",),
            signature_kwargs={},
            decorator_kwargs={
                "examples": ["Q: 1+1\nA: 2"],
                "payload": {"source": "unit-test"},
                "prompt": "Answer briefly.",
                "response_format": {"type": "json_object"},
                "suppress_verbose_output": True,
                "template_suffix": "answer",
            },
        )
        argument.prop.instance = SimpleNamespace(
            global_context=("Static facts.", "Dynamic facts."),
            _kwargs={},
        )
        argument.prop.processed_input = "What changed?"

        engine.prepare(argument)

        system = argument.prop.prepared_input[0]["content"]
        user = argument.prop.prepared_input[1]
        assert system.count("<meta_instruction>") == 1
        assert "</meta_instruction>" in system
        assert "<response_format>" in system
        assert "<type>json_object</type>" in system
        assert "<context>" in system
        assert "<static>" in system
        assert "Static facts." in system
        assert "</static>" in system
        assert "<dynamic>" in system
        assert "Dynamic facts." in system
        assert "</dynamic>" in system
        assert "<additional>" in system
        assert "source" in system
        assert "unit-test" in system
        assert "<examples>" in system
        assert "Q: 1+1" in system
        assert "<instruction>" in system
        assert "Answer briefly." in system
        assert "<template_suffix>" in system
        assert "<placeholder>answer</placeholder>" in system
        assert "<STATIC CONTEXT/>" not in system
        assert "<STATIC_CONTEXT/>" not in system
        assert user == {"role": "user", "content": "What changed?"}

    def test_neurosymbolic_self_prompt_template_uses_plain_output_examples(self):
        prompt = prompt_registry.render("chat.self_prompt")

        assert '{"system": "<new system prompt>", "user": "<new user prompt>"}' in prompt
        assert '{"developer": "<new developer prompt>", "user": "<new user prompt>"}' in prompt
        assert "<system_output>" not in prompt
        assert "<developer_output>" not in prompt
        assert "<variant>" not in prompt
        assert "<requirement>" not in prompt

    def test_engine_self_prompt_sends_prompt_object_as_raw_json(self):
        engine = self.make_engine(client_max_retries=0)
        response_json = self.mock_response_json()
        response_json["choices"][0]["message"] = {
            "role": "assistant",
            "content": json.dumps({"system": "new system prompt", "user": "new user prompt"}),
        }

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=response_json, request=request),
        ) as api:
            repository = EngineRepository()
            previous_engine = repository._engines.get("neurosymbolic")
            try:
                EngineRepository.register("neurosymbolic", engine, allow_engine_override=True)
                result = engine.self_prompt({"system": "old system", "user": "old user"})
            finally:
                if previous_engine is not None:
                    EngineRepository.register(
                        "neurosymbolic", previous_engine, allow_engine_override=True
                    )
                else:
                    repository._engines.pop("neurosymbolic", None)

        messages = api.last_body["messages"]
        assert result == {"system": "new system prompt", "user": "new user prompt"}
        assert messages[0]["role"] == "system"
        assert "Generate a new system or developer prompt" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert json.loads(messages[1]["content"]) == {"system": "old system", "user": "old user"}
        assert api.last_body["response_format"] == {"type": "json_object"}
