import base64
import json
from pathlib import Path
from typing import ClassVar

import pytest
import httpx

from symai.backend.engines.neurosymbolic.anthropic.engine import (
    ANTHROPIC_MESSAGES_URL,
    AnthropicEngine,
)
from symai.backend.engines.neurosymbolic.anthropic.models import (
    ANTHROPIC_MODEL_SPECS,
    ANTHROPIC_VERSION,
    API_PINNED,
    CACHE_CONTROL_1H,
    SUPPORTED_ANTHROPIC_MODELS,
    AnthropicResponse,
    anthropic_strip_prefix,
)
from symai.backend.engines.neurosymbolic.anthropic.stream import AnthropicStreamAdapter
from symai.backend.transport import SSEEvent
from symai.components import MetadataTracker
from symai.prompts import CACHE_BREAKPOINT
from tests.engines.mock_api import MockAPI
from tests.engines.neurosymbolic.interface import NeurosymbolicEngineTestInterface


class TestAnthropicEngine(NeurosymbolicEngineTestInterface):
    engine_cls = AnthropicEngine
    supported_models = tuple(SUPPORTED_ANTHROPIC_MODELS)
    model_specs = ANTHROPIC_MODEL_SPECS
    default_model = "anthropic:claude-haiku-4-5"
    response_cls = AnthropicResponse
    wire_provider = "anthropic"
    wire_operation = "messages.create"
    wire_url = ANTHROPIC_MESSAGES_URL
    supports_streaming = True
    api_pinned = API_PINNED
    cache_test_model = "anthropic:claude-sonnet-4-6"
    cache_unsupported_model_raises = False
    max_tokens_required = True
    supports_token_counting = True
    # NOTE: Anthropic streams by default (legacy contract); JSON-mock tests opt out.
    default_forward_kwargs: ClassVar[dict] = {"stream": False}

    def spec_for(self, model):
        return self.model_specs[anthropic_strip_prefix(model)]

    def expected_wire_model(self, model=None):
        return anthropic_strip_prefix(model or self.default_model)

    def assert_auth_headers(self, headers):
        assert headers["x-api-key"] == "sk-test-not-a-real-key"
        assert headers["anthropic-version"] == ANTHROPIC_VERSION

    def assert_cache_breakpoint_body(self, body, segments):
        texts = []
        for message in body["messages"]:
            content = message["content"]
            if isinstance(content, list):
                texts.extend(block for block in content if block.get("type") == "text")
        assert [block["text"] for block in texts] == segments
        for block in texts[:-1]:
            assert block["cache_control"] == CACHE_CONTROL_1H
        assert "cache_control" not in texts[-1]
        # block-level breakpoints supersede the top-level auto-cache form
        assert "cache_control" not in body

    def cache_write_tokens(self, usage):
        return usage.get("cache_creation_input_tokens") or 0

    def cache_read_tokens(self, usage):
        return usage.get("cache_read_input_tokens") or 0

    def usage_prompt_tokens(self, usage):
        return usage.get("input_tokens", usage.get("prompt_tokens"))

    def usage_completion_tokens(self, usage):
        return usage.get("output_tokens", usage.get("completion_tokens"))

    def usage_total_tokens(self, usage):
        return usage.get(
            "total_tokens", self.usage_prompt_tokens(usage) + self.usage_completion_tokens(usage)
        )

    def wire_input_expected(self, argument):
        # NOTE: Anthropic moves system to a top-level field; messages keep user/assistant.
        return [m for m in argument.prop.prepared_input if m["role"] != "system"]

    def assert_self_prompt_response_format(self, body):
        # NOTE: json_object without a schema produces no output_config on Anthropic.
        assert "output_config" not in body

    def assert_self_prompt_messages(self, body):
        # NOTE: Anthropic carries system top-level; messages hold user/assistant only.
        assert "Generate a new system or developer prompt" in body["system"]
        assert body["messages"][0]["role"] == "user"
        assert json.loads(body["messages"][0]["content"]) == {
            "system": "old system",
            "user": "old user",
        }

    def inject_self_prompt_response(self, payload, content):
        payload["content"] = [{"type": "text", "text": content}]
        return payload

    def mock_forward_response(self, request, payload):
        body = json.loads(request.content.decode())
        if not body.get("stream"):
            return httpx.Response(200, json=payload, request=request)
        text = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )
        return httpx.Response(
            200,
            content=self._sse_body(text),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    def vision_messages(self, image_path):
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode()
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded,
                        },
                    },
                    {"type": "text", "text": "What is in this image? Answer in one word."},
                ],
            }
        ]

    def mock_response_json(self):
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Add one and one."},
                {"type": "text", "text": "2"},
            ],
            "model": "claude-haiku-4-5",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 4,
            },
        }

    def response_dropping_content(self, payload):
        payload["content"] = []
        return payload

    def response_dropping_usage(self, payload):
        del payload["usage"]
        return payload

    def mock_sse_body(self):
        return self._sse_body("2", thinking="Add one and one.")

    def _sse_body(self, text, thinking=None):
        chunks = [
            (
                "message_start",
                {
                    "message": {
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    }
                },
            ),
        ]
        index = 0
        if thinking:
            chunks += [
                (
                    "content_block_start",
                    {"index": index, "content_block": {"type": "thinking", "thinking": ""}},
                ),
                (
                    "content_block_delta",
                    {"index": index, "delta": {"type": "thinking_delta", "thinking": thinking}},
                ),
                ("content_block_stop", {"index": index}),
            ]
            index += 1
        chunks += [
            (
                "content_block_start",
                {"index": index, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"index": index, "delta": {"type": "text_delta", "text": text}},
            ),
            ("content_block_stop", {"index": index}),
            (
                "message_delta",
                {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            ),
            ("message_stop", {}),
        ]
        lines = []
        for event, data in chunks:
            lines.append(f"event: {event}")
            lines.append(f"data: {json.dumps(data)}")
            lines.append("")
        return "\n".join(lines).encode("utf-8")

    def mock_tool_call_json(self):
        payload = self.mock_response_json()
        payload["content"] = [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"location": "Paris"},
            }
        ]
        return payload

    def weather_tool_spec(self):
        return {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }

    def tool_choice_kwarg(self):
        return {"tool_choice": {"type": "any"}}

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
        assert isinstance(raw_output, AnthropicResponse)
        assert raw_output.stop_reason == "end_turn"
        # usage merges message_start (input) and message_delta (output)
        assert raw_output.usage.input_tokens == 10
        assert raw_output.usage.output_tokens == 5

    def test_usage_tracking_includes_cache_breakdowns(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                engine.forward(self.make_prepared_argument(kwargs={"stream": False}))
            details = tracker.usage[(self.engine_cls.__name__, self.default_model)]

        assert details["usage"]["prompt_tokens"] == 10
        assert details["usage"]["completion_tokens"] == 5
        assert details["usage"]["total_tokens"] == 15
        assert details["prompt_breakdown"]["cached_tokens"] == 4
        assert details["extras"]["cache_creation_input_tokens"] == 2

    def test_build_request_strips_provider_prefix_from_wire_model(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert request.body()["model"] == "claude-haiku-4-5"

    def test_build_request_splits_system_out_of_messages(self):
        engine = self.make_engine()
        argument = self.make_query_argument("What is 1+1?")
        engine.prepare(argument)

        request = engine.build_request(argument)
        body = request.body()

        assert isinstance(body["system"], str) and body["system"]
        assert all(message["role"] != "system" for message in body["messages"])
        assert body["messages"][-1]["role"] == "user"

    def test_build_request_defaults_max_tokens_to_response_budget(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert request.body()["max_tokens"] == self.spec_for(self.default_model).response_tokens

    def test_build_request_streams_by_default(self):
        # NOTE: legacy contract — stream defaults to True because non-streamed requests
        # >10m error out at the API; users opt out with stream=False.
        request = self.make_engine().build_request(self.make_prepared_argument())
        assert request.body()["stream"] is True

        opt_out = self.make_engine().build_request(
            self.make_prepared_argument(kwargs={"stream": False})
        )
        assert opt_out.body()["stream"] is False

    def test_build_request_maps_json_schema_response_format_to_output_config(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        request = self.make_engine().build_request(
            self.make_prepared_argument(
                kwargs={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "Answer", "schema": schema},
                    }
                }
            )
        )

        assert request.body()["output_config"]["format"] == {
            "type": "json_schema",
            "schema": schema,
        }

    def test_build_request_json_object_response_format_emits_no_output_config(self):
        # NOTE: json_object mode is prompt-instructed on Anthropic; no wire field.
        request = self.make_engine().build_request(
            self.make_prepared_argument(kwargs={"response_format": {"type": "json_object"}})
        )

        assert "output_config" not in request.body()

    def test_build_request_treats_empty_stop_as_unset(self):
        engine = self.make_engine()

        empty_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ""}))
        assert "stop_sequences" not in empty_stop.body()

        user_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": "END"}))
        assert user_stop.body()["stop_sequences"] == ["END"]

    def test_build_request_strips_sampling_kwargs_for_opus_4_8(self):
        engine = self.make_engine(model="anthropic:claude-opus-4-8")

        request = engine.build_request(
            self.make_prepared_argument(kwargs={"temperature": 0.2, "top_p": 0.5, "top_k": 40})
        )
        body = request.body()

        assert "temperature" not in body
        assert "top_p" not in body
        assert "top_k" not in body

    def test_build_request_adaptive_thinking_config(self):
        engine = self.make_engine(model="anthropic:claude-opus-4-8")

        request = engine.build_request(
            self.make_prepared_argument(kwargs={"thinking": {"type": "adaptive", "effort": "high"}})
        )
        body = request.body()

        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "high"}

    def test_build_request_manual_thinking_config(self):
        engine = self.make_engine()

        request = engine.build_request(
            self.make_prepared_argument(
                kwargs={"thinking": {"type": "enabled", "budget_tokens": 2048}}
            )
        )

        assert request.body()["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    def test_build_request_cache_control_disabled_strips_marker(self):
        engine = self.make_engine()
        marked = [
            {"role": "user", "content": f"prefix {CACHE_BREAKPOINT} suffix"},
        ]

        body = engine.build_request(
            self.make_prepared_argument(messages=marked, kwargs={"cache_control": False})
        ).body()

        assert "cache_control" not in body
        assert body["messages"][0]["content"] == "prefix  suffix"

    def test_compute_required_tokens_uses_count_tokens_endpoint(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json={"input_tokens": 42}, request=request),
        ) as api:
            tokens = engine.compute_required_tokens([{"role": "user", "content": "hello"}])

        assert tokens == 42
        assert "count_tokens" in str(api.last_request.url)
        assert api.last_request.headers["x-api-key"] == "sk-test-not-a-real-key"
        assert api.last_request.headers["anthropic-version"] == ANTHROPIC_VERSION

    def test_build_request_preserves_system_content_blocks(self):
        system_blocks = [
            {"type": "text", "text": "brief"},
            {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}},
        ]
        argument = self.make_prepared_argument(
            messages=[
                {"role": "system", "content": system_blocks},
                {"role": "user", "content": "go"},
            ]
        )

        body = self.make_engine().build_request(argument).body()

        assert body["system"] == system_blocks
        assert body["messages"] == [{"role": "user", "content": "go"}]

    def test_complete_posts_payload_without_rewriting(self):
        engine = self.make_engine(model="anthropic:claude-opus-5")
        system = [
            {"type": "text", "text": "brief"},
            {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}},
        ]
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": "sig"},
                    {"type": "tool_use", "id": "toolu_1", "name": "list_assets", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ]
        payload = {
            "model": "claude-opus-5",
            "max_tokens": 256,
            "system": system,
            "tools": [self.weather_tool_spec()],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
            "messages": messages,
            "stream": False,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ) as api:
            response = engine.complete(payload)

        body = api.last_body
        assert body["system"] == system
        assert body["messages"] == messages
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "low"}
        assert body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert isinstance(response, AnthropicResponse)
        assert response.content[0].type == "thinking"

    def test_complete_stream_keeps_signature_and_parallel_tools(self):
        engine = self.make_engine()

        with MockAPI(
            engine,
            lambda request: httpx.Response(
                200,
                content=_roundtrip_sse_body(),
                headers={"content-type": "text/event-stream"},
                request=request,
            ),
        ):
            response = engine.complete(
                {
                    "messages": [{"role": "user", "content": "read the assets"}],
                    "max_tokens": 64,
                    "stream": True,
                }
            )

        types = [block.type for block in response.content]
        assert types == ["thinking", "text", "tool_use", "tool_use"]
        assert response.content[0].thinking == "Need both files."
        assert response.content[0].signature == "sig-abc"
        assert response.content[1].text == "calling tools"
        assert response.content[2].name == "list_assets"
        assert response.content[2].input == {}
        assert response.content[3].name == "read_asset"
        assert response.content[3].input == {"asset_id": "a1"}
        assert response.stop_reason == "tool_use"

    def test_complete_forwards_unknown_messages_fields(self):
        # New API params between payload pins must not be eaten by the strict model.
        engine = self.make_engine()

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ) as api:
            engine.complete(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "stream": False,
                    "container": {"id": "ctx_123"},
                    "service_tier": "standard",
                }
            )

        body = api.last_body
        assert body["container"] == {"id": "ctx_123"}
        assert body["service_tier"] == "standard"

    @pytest.mark.engine_live
    def test_live_complete_roundtrip_opus_5(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)
        engine = self.make_live_engine("anthropic:claude-opus-5", api_key)
        system = [
            {"type": "text", "text": "Answer briefly."},
            {
                "type": "text",
                "text": "Use only the requested words.",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        user = {"role": "user", "content": "What is 37 times 41? Reply with only the number."}
        first = engine.complete(
            {
                "model": "claude-opus-5",
                "max_tokens": 4096,
                "system": system,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "messages": [user],
            },
            timeout=120.0,
        )
        thinking = [block for block in first.content if block.type == "thinking"]
        text = [block for block in first.content if block.type == "text"]
        assert thinking, f"expected thinking; types={[b.type for b in first.content]}"
        assert thinking[0].signature, "streamed thinking must keep its signature"
        assert text and "1517" in (text[0].text or "")

        second = engine.complete(
            {
                "model": "claude-opus-5",
                "max_tokens": 4096,
                "system": system,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "messages": [
                    user,
                    {
                        "role": "assistant",
                        "content": [
                            block.model_dump(exclude_none=True) for block in first.content
                        ],
                    },
                    {"role": "user", "content": "Reply with exactly: confirmed"},
                ],
            },
            timeout=120.0,
        )

        assert second.stop_reason == "end_turn"
        assert second.usage.input_tokens > 0
        confirmed = "".join(block.text or "" for block in second.content if block.type == "text")
        assert "confirmed" in confirmed.lower()

    @pytest.mark.engine_live
    def test_live_complete_tool_roundtrip_opus_5(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)
        engine = self.make_live_engine("anthropic:claude-opus-5", api_key)
        tools = [self.weather_tool_spec()]
        user = {
            "role": "user",
            "content": "What is the weather in Paris? Call get_weather. Do not guess.",
        }
        first = engine.complete(
            {
                "model": "claude-opus-5",
                "max_tokens": 4096,
                "tools": tools,
                "tool_choice": {"type": "any"},
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "messages": [user],
            },
            timeout=120.0,
        )
        tool_uses = [block for block in first.content if block.type == "tool_use"]
        thinking = [block for block in first.content if block.type == "thinking"]
        assert tool_uses, f"expected tool_use; types={[b.type for b in first.content]}"
        assert tool_uses[0].name == "get_weather"
        assert tool_uses[0].id
        if thinking:
            assert thinking[0].signature, "thinking next to a tool call must keep its signature"

        results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "18C, clear",
            }
            for block in tool_uses
        ]
        second = engine.complete(
            {
                "model": "claude-opus-5",
                "max_tokens": 4096,
                "tools": tools,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "messages": [
                    user,
                    {
                        "role": "assistant",
                        "content": [
                            block.model_dump(exclude_none=True) for block in first.content
                        ],
                    },
                    {"role": "user", "content": results},
                ],
            },
            timeout=120.0,
        )

        assert second.stop_reason in {"end_turn", "tool_use"}
        assert second.usage.input_tokens > 0
        if second.stop_reason == "end_turn":
            answer = "".join(block.text or "" for block in second.content if block.type == "text")
            assert "18" in answer


def _sse_event(event, data):
    return SSEEvent(event=event, data=json.dumps(data))


def _roundtrip_sse_body():
    chunks = [
        (
            "message_start",
            {"message": {"role": "assistant", "usage": {"input_tokens": 10, "output_tokens": 1}}},
        ),
        (
            "content_block_start",
            {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "thinking_delta", "thinking": "Need both files."}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc"}},
        ),
        ("content_block_stop", {"index": 0}),
        ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
        (
            "content_block_delta",
            {"index": 1, "delta": {"type": "text_delta", "text": "calling tools"}},
        ),
        ("content_block_stop", {"index": 1}),
        (
            "content_block_start",
            {
                "index": 2,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "list_assets"},
            },
        ),
        (
            "content_block_delta",
            {"index": 2, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        ),
        ("content_block_stop", {"index": 2}),
        (
            "content_block_start",
            {
                "index": 3,
                "content_block": {"type": "tool_use", "id": "toolu_2", "name": "read_asset"},
            },
        ),
        (
            "content_block_delta",
            {
                "index": 3,
                "delta": {"type": "input_json_delta", "partial_json": '{"asset_id":'},
            },
        ),
        (
            "content_block_delta",
            {"index": 3, "delta": {"type": "input_json_delta", "partial_json": ' "a1"}'}},
        ),
        ("content_block_stop", {"index": 3}),
        (
            "message_delta",
            {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}},
        ),
        ("message_stop", {}),
    ]
    lines = []
    for event, data in chunks:
        lines.append(f"event: {event}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")


class TestAnthropicStreamAdapter:
    def test_keeps_signature_and_parallel_tool_blocks(self):
        adapter = AnthropicStreamAdapter()
        for event, data in [
            (
                "message_start",
                {"message": {"usage": {"input_tokens": 3, "output_tokens": 1}}},
            ),
            (
                "content_block_start",
                {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "thinking_delta", "thinking": "x"}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "signature_delta", "signature": "s1"}},
            ),
            ("content_block_stop", {"index": 0}),
            (
                "content_block_start",
                {
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "list_assets"},
                },
            ),
            (
                "content_block_delta",
                {"index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
            ),
            ("content_block_stop", {"index": 1}),
            (
                "content_block_start",
                {
                    "index": 2,
                    "content_block": {"type": "tool_use", "id": "t2", "name": "read_asset"},
                },
            ),
            (
                "content_block_delta",
                {"index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"id":1}'}},
            ),
            ("content_block_stop", {"index": 2}),
            ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}}),
            ("message_stop", {}),
        ]:
            adapter.process_event(_sse_event(event, data))

        content = adapter.content()
        assert [block["type"] for block in content] == ["thinking", "tool_use", "tool_use"]
        assert content[0]["signature"] == "s1"
        assert content[1]["name"] == "list_assets"
        assert content[2]["input"] == {"id": 1}
        assert [call["name"] for call in adapter.tool_calls] == ["list_assets", "read_asset"]

    def test_keeps_redacted_thinking_data(self):
        adapter = AnthropicStreamAdapter()
        for event, data in [
            ("message_start", {"message": {"usage": {"input_tokens": 3, "output_tokens": 1}}}),
            (
                "content_block_start",
                {
                    "index": 0,
                    "content_block": {"type": "redacted_thinking", "data": "encrypted-payload"},
                },
            ),
            ("content_block_stop", {"index": 0}),
            ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "ok"}}),
            ("content_block_stop", {"index": 1}),
            (
                "message_delta",
                {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
            ),
            ("message_stop", {}),
        ]:
            adapter.process_event(_sse_event(event, data))

        content = adapter.content()
        assert content[0]["type"] == "redacted_thinking"
        assert content[0]["data"] == "encrypted-payload"

    def test_accumulates_server_tool_use_input_but_keeps_it_off_tool_calls(self):
        adapter = AnthropicStreamAdapter()
        for event, data in [
            ("message_start", {"message": {"usage": {"input_tokens": 3, "output_tokens": 1}}}),
            (
                "content_block_start",
                {
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"query":"x"}'}},
            ),
            ("content_block_stop", {"index": 0}),
            ("message_delta", {"delta": {"stop_reason": "pause_turn"}, "usage": {"output_tokens": 2}}),
            ("message_stop", {}),
        ]:
            adapter.process_event(_sse_event(event, data))

        content = adapter.content()
        assert content[0]["input"] == {"query": "x"}
        assert adapter.tool_calls == []

    def test_server_tool_result_block_keeps_tool_use_id_and_content(self):
        from symai.backend.engines.neurosymbolic.anthropic.models import AnthropicContentBlock

        block = AnthropicContentBlock.model_validate(
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [{"type": "web_search_result", "url": "https://example.com"}],
            }
        )

        dumped = block.model_dump(exclude_none=True)
        assert dumped["tool_use_id"] == "srvtoolu_1"
        assert dumped["content"] == [{"type": "web_search_result", "url": "https://example.com"}]
