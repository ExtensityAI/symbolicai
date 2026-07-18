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
import uuid
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
from symai.prompts import CACHE_BREAKPOINT

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
    # NOTE: the wire key carrying the max-tokens kwarg differs per provider
    # (e.g. Groq remaps max_tokens -> max_completion_tokens at the engine).
    max_tokens_wire_key: ClassVar[str] = "max_tokens"
    # NOTE: the kwarg name users pass for max tokens (OpenAI accepts max_output_tokens
    # natively; chat-completions providers accept max_tokens).
    request_max_tokens_kwarg: ClassVar[str] = "max_tokens"
    # NOTE: the request-body key carrying the prepared input ("messages" for chat
    # completions, "input" for the Responses API).
    wire_input_key: ClassVar[str] = "messages"
    # NOTE: stream options the engine must force on streaming requests; None when the
    # provider's streams always carry usage (Responses API) or streaming is unsupported.
    stream_options_expected: ClassVar = {"include_usage": True}
    # NOTE: model used by the explicit-cache tests; None when the provider has no
    # cache-breakpoint support (deepseek/cerebras/groq auto-cache only). The cache
    # suite activates per provider by setting this (anthropic once migrated).
    cache_test_model: ClassVar[str | None] = None
    # NOTE: whether a marker on a non-supporting model raises (OpenAI semantics).
    # Anthropic caches on every model by default, so its test asserts stripping instead.
    cache_unsupported_model_raises: ClassVar[bool] = True
    # NOTE: whether the provider's API requires max_tokens on every request (Anthropic)
    # — the engine then defaults it to the model's response budget instead of omitting.
    max_tokens_required: ClassVar[bool] = False
    # NOTE: whether the engine implements compute_required_tokens (openai/anthropic via
    # tiktoken or the count endpoint; deepseek/groq/openrouter raise NotImplementedError).
    supports_token_counting: ClassVar[bool] = False

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

    def usage_prompt_tokens(self, usage: dict) -> int:
        """Prompt/input token count in the provider's usage shape."""
        return usage["prompt_tokens"]

    def wire_input_expected(self, argument) -> list:
        """The prepared input as it appears on the wire (Anthropic splits system out)."""
        return argument.prop.prepared_input

    def usage_completion_tokens(self, usage: dict) -> int:
        """Completion/output token count in the provider's usage shape."""
        return usage["completion_tokens"]

    def usage_total_tokens(self, usage: dict) -> int:
        """Total token count in the provider's usage shape."""
        return usage["total_tokens"]

    def inject_self_prompt_response(self, payload: dict, content: str) -> dict:
        """Place the self-prompt JSON answer into the provider's mock response shape."""
        payload["choices"][0]["message"] = {"role": "assistant", "content": content}
        return payload

    def assert_self_prompt_response_format(self, body: dict):
        """What the provider does with the response_format kwarg on self-prompt calls."""
        assert body["response_format"] == {"type": "json_object"}

    def assert_self_prompt_messages(self, body: dict):
        """The self-prompt request's system + user structure on the wire."""
        messages = body[self.wire_input_key]
        assert messages[0]["role"] == "system"
        assert "Generate a new system or developer prompt" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert json.loads(messages[1]["content"]) == {"system": "old system", "user": "old user"}

    def assert_auth_headers(self, headers: dict):
        """Provider auth wire convention (Bearer for most, x-api-key for Anthropic)."""
        auth = headers.get("authorization") or headers.get("Authorization")
        assert auth == f"Bearer {DUMMY_KEY}"

    def assert_cache_breakpoint_body(self, body: dict, segments: list[str]):
        """Assert the provider transformed a two-segment marked prompt into cache
        blocks honoring its wire convention (OpenAI: prompt_cache_breakpoint on the
        first block; Anthropic: cache_control blocks once migrated)."""
        raise NotImplementedError

    def cache_write_tokens(self, usage: dict) -> int:
        """Tokens written to cache per the provider's usage shape."""
        raise NotImplementedError

    def cache_read_tokens(self, usage: dict) -> int:
        """Tokens read from cache per the provider's usage shape."""
        raise NotImplementedError

    def mock_tool_call_json(self) -> dict:
        """Provider-shaped response containing one get_weather tool call."""
        raise NotImplementedError

    def weather_tool_spec(self) -> dict:
        """The provider's get_weather tool definition for the live tool smoke."""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }

    def tool_choice_kwarg(self) -> dict:
        """Kwarg forcing the tool call (Anthropic uses 'any', others 'required')."""
        return {"tool_choice": "required"}

    # --- shared helpers ---
    def spec_for(self, model: str):
        """Resolve a (possibly provider-prefixed) model name to its spec."""
        return self.model_specs[model]

    def expected_wire_model(self, model: str | None = None) -> str:
        """The model id expected on the wire (override when config names carry prefixes)."""
        return model or self.default_model

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
        pricing = self.spec_for(model).pricing
        assert pricing is not None, f"no published pricing for {model} at API_PINNED"
        if pricing.cached_input is not None:
            input_cost = (
                usage.get("prompt_cache_hit_tokens", 0) * pricing.cached_input
                + usage.get("prompt_cache_miss_tokens", self.usage_prompt_tokens(usage))
                * pricing.input
            )
        else:
            input_cost = self.usage_prompt_tokens(usage) * pricing.input
        return (input_cost + self.usage_completion_tokens(usage) * pricing.output) / 1_000_000

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
        for model in self.supported_models:
            spec = self.spec_for(model)
            assert isinstance(spec.reasoning, bool)
            assert isinstance(spec.vision, bool)
            assert spec.context_tokens > 0
            assert spec.response_tokens > 0
            # NOTE: pricing may be None when the provider has not published per-token
            # prices for the model at API_PINNED (e.g. preview models).
            if spec.pricing is not None:
                assert spec.pricing.input >= 0
                assert spec.pricing.output >= 0
                if spec.pricing.cached_input is not None:
                    assert spec.pricing.cached_input <= spec.pricing.input

    def test_build_request_wire_shape(self):
        engine = self.make_engine(client_timeout=7.0)
        argument = self.make_prepared_argument(
            kwargs={
                "temperature": 0.2,
                self.request_max_tokens_kwarg: 32,
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
        self.assert_auth_headers(request.headers)
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["X-Test"] == "1"
        assert request.params == {"debug": "1"}
        assert request.timeout == 7.0

        assert body["model"] == self.expected_wire_model()
        assert body["temperature"] == 0.2
        assert body[self.max_tokens_wire_key] == 32
        assert body["vendor_flag"] is True
        assert "extra_body" not in body
        assert "extra_headers" not in body
        assert "extra_query" not in body

    def test_build_request_omits_max_tokens_when_not_provided(self):
        request = self.make_engine().build_request(self.make_prepared_argument())
        body = request.body()

        if self.max_tokens_required:
            assert (
                body[self.max_tokens_wire_key] == self.spec_for(self.default_model).response_tokens
            )
        else:
            assert "max_tokens" not in body
            assert "max_completion_tokens" not in body

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
            engine.build_request(
                self.make_prepared_argument(kwargs={self.request_max_tokens_kwarg: "32"})
            )

        with pytest.raises(ValueError, match="Unsupported request kwargs"):
            engine.build_request(
                self.make_prepared_argument(
                    kwargs={"strict_request_kwargs": True, "temprature": 0.2}
                )
            )

    def test_forward_mock_transport_returns_typed_response(self):
        engine = self.make_engine(client_max_retries=0)
        argument = self.make_query_argument(
            "What is 1+1?", **{self.request_max_tokens_kwarg: 16}, extra_query={"debug": "1"}
        )

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ) as api:
            engine.prepare(argument)
            output, metadata = engine.forward(argument)

        assert api.last_request.method == "POST"
        assert str(api.last_request.url) == f"{self.wire_url}?debug=1"
        self.assert_auth_headers(dict(api.last_request.headers))
        assert api.last_body["model"] == self.expected_wire_model()
        assert api.last_body[self.max_tokens_wire_key] == 16
        assert api.last_body[self.wire_input_key] == self.wire_input_expected(argument)
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
        assert details["usage"]["prompt_tokens"] == 2 * self.usage_prompt_tokens(mock_usage)
        assert details["usage"]["completion_tokens"] == 2 * self.usage_completion_tokens(mock_usage)
        assert details["usage"]["total_tokens"] == 2 * self.usage_total_tokens(mock_usage)
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

    def test_cache_marker_never_reaches_wire_unsupported(self):
        if self.cache_test_model is not None:
            pytest.skip(f"{self.engine_cls.__name__} honors cache breakpoints")

        engine = self.make_engine()
        marked = [{"role": "user", "content": f"first part {CACHE_BREAKPOINT} second part"}]

        body = engine.build_request(self.make_prepared_argument(messages=marked)).body()

        assert CACHE_BREAKPOINT not in json.dumps(body)
        assert "first part  second part" in json.dumps(
            body
        ) or "first part second part" in json.dumps(body)

    def test_cache_breakpoint_blocks_built(self):
        if self.cache_test_model is None:
            pytest.skip(f"{self.engine_cls.__name__} has no explicit cache support")

        engine = self.make_engine(model=self.cache_test_model)
        marked = [{"role": "user", "content": f"cached prefix {CACHE_BREAKPOINT} fresh suffix"}]

        body = engine.build_request(self.make_prepared_argument(messages=marked)).body()

        self.assert_cache_breakpoint_body(body, ["cached prefix ", " fresh suffix"])

    def test_cache_breakpoint_rejects_unsupported_model(self):
        if self.cache_test_model is None:
            pytest.skip(f"{self.engine_cls.__name__} has no explicit cache support")
        if not self.cache_unsupported_model_raises:
            pytest.skip(f"{self.engine_cls.__name__} strips markers instead of raising")

        engine = self.make_engine()
        marked = [{"role": "user", "content": f"prefix {CACHE_BREAKPOINT} suffix"}]

        with pytest.raises(ValueError, match="cache"):
            engine.build_request(self.make_prepared_argument(messages=marked))

    def test_cache_breakpoint_rejects_too_many(self):
        if self.cache_test_model is None:
            pytest.skip(f"{self.engine_cls.__name__} has no explicit cache support")

        engine = self.make_engine(model=self.cache_test_model)
        marked = [
            {"role": "user", "content": CACHE_BREAKPOINT.join(["a", "b", "c", "d", "e", "f"])}
        ]

        with pytest.raises(ValueError, match="at most"):
            engine.build_request(self.make_prepared_argument(messages=marked))

    def test_cache_breakpoint_rejects_empty_segment(self):
        if self.cache_test_model is None:
            pytest.skip(f"{self.engine_cls.__name__} has no explicit cache support")

        engine = self.make_engine(model=self.cache_test_model)
        marked = [{"role": "user", "content": f"{CACHE_BREAKPOINT} suffix"}]

        with pytest.raises(ValueError, match="non-empty"):
            engine.build_request(self.make_prepared_argument(messages=marked))

    def test_tool_call_extraction(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_tool_call_json(), request=request),
        ):
            _output, metadata = engine.forward(self.make_prepared_argument())

        function_call = metadata.get("function_call")
        assert function_call is not None
        assert function_call["name"] == "get_weather"
        assert isinstance(function_call["arguments"], dict)
        assert "location" in function_call["arguments"]

    @pytest.mark.engine_live
    def test_live_tool_smoke(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(self.default_model, api_key)
        messages = [
            {
                "role": "user",
                "content": "What is the weather in Paris? Use the get_weather tool.",
            }
        ]
        kwargs = {"tools": [self.weather_tool_spec()], **self.tool_choice_kwarg()}

        output, metadata = engine.forward(
            self.make_prepared_argument(kwargs=kwargs, messages=messages)
        )

        function_call = metadata.get("function_call")
        assert function_call is not None, f"no tool call; output was: {output!r}"
        assert function_call["name"] == "get_weather"
        assert "location" in function_call["arguments"]

    @pytest.mark.engine_live
    def test_live_token_count_matches_generation_usage(self, engine_api_mode):
        if not self.supports_token_counting:
            pytest.skip(f"{self.engine_cls.__name__} has no token counting")
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(self.default_model, api_key)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "New synergies will help drive top-line growth."},
            {"role": "assistant", "content": "Things working well together will increase revenue."},
            {"role": "user", "content": "Let's circle back when we have more bandwidth."},
        ]

        estimated = engine.compute_required_tokens(messages)
        _output, metadata = engine.forward(self.make_prepared_argument(messages=messages))
        actual = self.usage_prompt_tokens(metadata["raw_output"].usage.model_dump())

        # NOTE: provider counters and local estimates may drift a few tokens per message;
        # the contract is closeness, not identity (Anthropic's endpoint is exact).
        assert abs(estimated - actual) <= max(4, int(0.1 * actual)), (
            f"estimated {estimated} vs actual {actual}"
        )

    @pytest.mark.engine_live
    def test_live_cache_write_then_read(self, engine_api_mode):
        if self.cache_test_model is None:
            pytest.skip(f"{self.engine_cls.__name__} has no explicit cache support")
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(self.cache_test_model, api_key)
        # NOTE: provider caching engages only above a minimum prompt size; ~1500 tokens
        # of prefix clears OpenAI's 1024-token floor for cache writes. A nonce prefix
        # keeps the first call cold across runs (cache life is ~30 min), otherwise a
        # warm cache turns the expected write into a read and flakes the assertion.
        nonce = uuid.uuid4().hex
        prefix = f"{nonce} lorem ipsum dolor sit amet " * 300
        marked = [{"role": "user", "content": f"{prefix}{CACHE_BREAKPOINT} Reply with exactly: ok"}]

        first = engine.forward(self.make_prepared_argument(messages=marked))[1]
        first_writes = self.cache_write_tokens(first["raw_output"].usage.model_dump())
        assert first_writes > 0, "first call should write cache"

        second = engine.forward(self.make_prepared_argument(messages=marked))[1]
        reads = self.cache_read_tokens(second["raw_output"].usage.model_dump())
        assert reads > 0, "second call should read from cache"

    @pytest.mark.engine_live
    def test_live_smoke(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        for model in self.supported_models:
            engine = self.make_live_engine(model, api_key)
            # NOTE: reasoning models spend budget on thinking first; 128 tokens is the
            # cheap floor that still guarantees visible content on trivial prompts.
            argument = self.make_query_argument(LIVE_PROMPT, **{self.request_max_tokens_kwarg: 128})

            engine.prepare(argument)
            output, metadata = engine.forward(argument)

            assert isinstance(output[0], str) and output[0]
            raw_output = metadata["raw_output"]
            assert isinstance(raw_output, self.response_cls)
            assert self.usage_total_tokens(raw_output.usage.model_dump()) > 0
            if self.spec_for(model).pricing is not None:
                cost = self.expected_cost_usd(model, raw_output.usage.model_dump())
                assert 0 < cost < 0.01

    @pytest.mark.engine_live
    def test_live_stream_smoke(self, engine_api_mode):
        if not self.supports_streaming:
            pytest.skip(f"{self.engine_cls.__name__} does not stream")
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(self.default_model, api_key)
        argument = self.make_query_argument(
            LIVE_PROMPT, **{self.request_max_tokens_kwarg: 128}, stream=True
        )

        engine.prepare(argument)
        output, metadata = engine.forward(argument)

        assert isinstance(output[0], str)
        assert self.usage_total_tokens(metadata["raw_output"].usage.model_dump()) > 0

    @pytest.mark.engine_live
    def test_live_vision(self, engine_api_mode):
        spec = self.spec_for(self.default_model)
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
        response_json = self.inject_self_prompt_response(
            self.mock_response_json(),
            json.dumps({"system": "new system prompt", "user": "new user prompt"}),
        )

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

        assert result == {"system": "new system prompt", "user": "new user prompt"}
        self.assert_self_prompt_messages(api.last_body)
        self.assert_self_prompt_response_format(api.last_body)
