import base64
import json
from pathlib import Path

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
from symai.components import MetadataTracker
from symai.prompts import CACHE_BREAKPOINT
from tests.engines.interface import MockAPI, NeurosymbolicEngineTestInterface


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
            (
                "content_block_start",
                {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "thinking_delta", "thinking": "Add one "}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "thinking_delta", "thinking": "and one."}},
            ),
            ("content_block_stop", {"index": 0}),
            ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "2"}}),
            ("content_block_stop", {"index": 1}),
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
                engine.forward(self.make_prepared_argument())
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
