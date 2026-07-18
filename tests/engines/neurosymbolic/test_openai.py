import base64
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines.neurosymbolic.openai.engine import (
    OPENAI_RESPONSES_URL,
    OpenAIEngine,
)
from symai.backend.engines.neurosymbolic.openai.models import (
    API_PINNED,
    OPENAI_MODEL_SPECS,
    SUPPORTED_OPENAI_MODELS,
    OpenAIResponse,
    openai_strip_prefix,
)
from symai.components import MetadataTracker
from tests.engines.mock_api import MockAPI
from tests.engines.neurosymbolic.interface import NeurosymbolicEngineTestInterface


class TestOpenAIEngine(NeurosymbolicEngineTestInterface):
    engine_cls = OpenAIEngine
    supported_models = tuple(SUPPORTED_OPENAI_MODELS)
    model_specs = OPENAI_MODEL_SPECS
    # NOTE: gpt-4.1-mini is the non-reasoning default so the generic wire-shape checks
    # (temperature passthrough, system role) hold; reasoning behavior is tested explicitly.
    default_model = "openai:gpt-4.1-mini"
    response_cls = OpenAIResponse
    wire_provider = "openai"
    wire_operation = "responses.create"
    wire_url = OPENAI_RESPONSES_URL
    supports_streaming = True
    api_pinned = API_PINNED
    request_max_tokens_kwarg = "max_output_tokens"
    max_tokens_wire_key = "max_output_tokens"
    wire_input_key = "input"
    stream_options_expected = None
    cache_test_model = "openai:gpt-5.6-terra"
    supports_token_counting = True

    def spec_for(self, model):
        return self.model_specs[openai_strip_prefix(model)]

    def usage_prompt_tokens(self, usage: dict) -> int:
        return usage.get("input_tokens", usage.get("prompt_tokens"))

    def usage_completion_tokens(self, usage: dict) -> int:
        return usage.get("output_tokens", usage.get("completion_tokens"))

    def inject_self_prompt_response(self, payload, content):
        for item in payload["output"]:
            if item["type"] == "message":
                item["content"] = [{"type": "output_text", "text": content}]
        return payload

    def assert_self_prompt_response_format(self, body: dict):
        # NOTE: the Responses API has no response_format field; the kwarg is dropped.
        assert "response_format" not in body

    def assert_cache_breakpoint_body(self, body, segments):
        texts = []
        for message in body["input"]:
            content = message["content"]
            if isinstance(content, list):
                texts.extend(block for block in content if block.get("type") == "input_text")
        assert [block["text"] for block in texts] == segments
        assert texts[0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
        assert "prompt_cache_breakpoint" not in texts[-1]

    def cache_write_tokens(self, usage: dict) -> int:
        return (usage.get("input_tokens_details") or {}).get("cache_write_tokens") or 0

    def cache_read_tokens(self, usage: dict) -> int:
        return (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0

    def expected_wire_model(self, model=None):
        return openai_strip_prefix(model or self.default_model)

    def vision_messages(self, image_path):
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode()
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"},
                    {"type": "input_text", "text": "What is in this image? Answer in one word."},
                ],
            }
        ]

    def mock_response_json(self):
        return {
            "id": "resp-test",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Add one and one."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "2", "annotations": []}],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
                "output_tokens_details": {"reasoning_tokens": 3},
            },
        }

    def response_dropping_content(self, payload):
        for item in payload["output"]:
            if item["type"] == "message":
                item["content"] = None
        return payload

    def response_dropping_usage(self, payload):
        del payload["usage"]
        return payload

    def mock_sse_body(self):
        chunks = [
            ("response.created", {"response": {"status": "in_progress"}}),
            (
                "response.reasoning_summary_text.delta",
                {"delta": "Add one and one."},
            ),
            ("response.output_text.delta", {"delta": "2"}),
            (
                "response.completed",
                {
                    "response": {
                        "status": "completed",
                        "output": [
                            {
                                "type": "reasoning",
                                "summary": [{"type": "summary_text", "text": "Add one and one."}],
                            },
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "2"}],
                            },
                        ],
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                },
            ),
        ]
        lines = []
        for event, data in chunks:
            lines.append(f"event: {event}")
            lines.append(f"data: {json.dumps(data)}")
            lines.append("")
        return "\n".join(lines).encode("utf-8")

    def mock_tool_call_json(self):
        payload = self.mock_response_json()
        payload["output"] = [
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
                "call_id": "call_1",
            }
        ]
        return payload

    def make_reasoning_engine(self, model="openai:gpt-5.4-mini", **kwargs):
        return self.make_engine(model=model, **kwargs)

    def test_forward_streams_sse_and_aggregates_response(self):
        engine = self.make_reasoning_engine(client_max_retries=0)

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
        assert isinstance(raw_output, OpenAIResponse)
        assert raw_output.status == "completed"
        assert raw_output.usage.total_tokens == 15

    def test_stream_fails_fast_when_completed_event_is_missing(self):
        engine = self.make_engine(client_max_retries=0)
        body = b'event: response.output_text.delta\ndata: {"delta": "2"}\n\n'

        with (
            MockAPI(
                engine,
                lambda request: httpx.Response(
                    200,
                    content=body,
                    headers={"content-type": "text/event-stream"},
                    request=request,
                ),
            ),
            pytest.raises(ValidationError),
        ):
            # the fallback reassembly has no usage without a terminal completed event
            engine.forward(self.make_prepared_argument(kwargs={"stream": True}))

    def test_parse_response_fails_on_non_completed_status(self):
        engine = self.make_engine(client_max_retries=0)
        payload = self.mock_response_json()
        payload["status"] = "failed"
        payload["error"] = {"code": "server_error", "message": "boom"}

        with (
            MockAPI(
                engine,
                lambda request: httpx.Response(200, json=payload, request=request),
            ),
            pytest.raises(ValueError, match="failed"),
        ):
            engine.forward(self.make_prepared_argument())

    def test_build_request_strips_sampling_kwargs_for_reasoning_models(self):
        engine = self.make_reasoning_engine()

        request = engine.build_request(
            self.make_prepared_argument(kwargs={"temperature": 0.2, "top_p": 0.5})
        )
        body = request.body()

        assert "temperature" not in body
        assert "top_p" not in body
        assert body["reasoning"] == {"effort": "medium"}

    def test_build_request_defaults_high_effort_for_pro_models(self):
        engine = self.make_reasoning_engine(model="openai:gpt-5.5-pro")

        request = engine.build_request(self.make_prepared_argument())

        assert request.body()["reasoning"] == {"effort": "high"}

    def test_prepare_uses_developer_role_for_reasoning_models(self):
        engine = self.make_reasoning_engine()
        argument = self.make_query_argument("What is 1+1?")

        engine.prepare(argument)

        assert argument.prop.prepared_input[0]["role"] == "developer"

    def test_build_request_converts_chat_function_tools(self):
        engine = self.make_engine()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ]

        request = engine.build_request(self.make_prepared_argument(kwargs={"tools": tools}))
        body = request.body()

        assert body["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
            }
        ]
        assert body["tool_choice"] == "auto"

    def test_build_request_clamps_max_output_tokens_with_warning(self):
        engine = self.make_engine()
        huge = self.spec_for(self.default_model).response_tokens + 100_000

        request = engine.build_request(
            self.make_prepared_argument(kwargs={"max_output_tokens": huge})
        )

        assert request.body()["max_output_tokens"] < huge

    def test_prepare_maps_vision_markers_to_input_image_parts(self):
        engine = self.make_engine()
        argument = self.make_query_argument("<<vision:tests/data/sample.jpg:>> What is this?")

        engine.prepare(argument)

        user = argument.prop.prepared_input[1]
        assert user["role"] == "user"
        assert user["content"][0]["type"] == "input_image"
        assert user["content"][0]["image_url"].startswith("data:image/jpeg;base64,")
        assert user["content"][1] == {"type": "input_text", "text": " What is this?"}

    def test_usage_tracking_includes_cache_and_reasoning_breakdowns(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                engine.forward(self.make_prepared_argument(kwargs={"max_output_tokens": 16}))
            details = tracker.usage[(self.engine_cls.__name__, self.default_model)]

        assert details["prompt_breakdown"]["cached_tokens"] == 4
        assert details["completion_breakdown"]["reasoning_tokens"] == 3
        assert details["extras"]["cache_write_tokens"] == 2
