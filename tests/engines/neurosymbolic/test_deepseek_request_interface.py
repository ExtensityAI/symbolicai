import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines.neurosymbolic.engine_deepseekX import (
    DEEPSEEK_CHAT_COMPLETIONS_URL,
    DeepSeekXReasoningEngine,
)
from symai.backend.engines.neurosymbolic.prompts import prompt_registry
from symai.backend.mixin.deepseek import DEEPSEEK_MODEL_SPECS, SUPPORTED_MODELS, DeepSeekResponse
from symai.backend.settings import SYMAI_CONFIG
from symai.components import MetadataTracker
from symai.core import Argument
from symai.functional import EngineRepository

DUMMY_KEY = "sk-test-not-a-real-key"


def make_engine(model="deepseek-v4-flash", **kwargs):
    return DeepSeekXReasoningEngine(api_key=DUMMY_KEY, model=model, **kwargs)


def make_prepared_argument(kwargs=None, messages=None):
    if kwargs is None:
        kwargs = {}
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return SimpleNamespace(kwargs=kwargs, prop=SimpleNamespace(prepared_input=messages))


def deepseek_response_json(content="2", reasoning_content="Add one and one."):
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


def sse_stream_body(chunks):
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def test_deepseek_supported_models_track_capabilities():
    for spec in DEEPSEEK_MODEL_SPECS.values():
        assert spec.reasoning is True
        assert spec.vision is False


def test_deepseek_id_requires_supported_model_and_api_key():
    engine = DeepSeekXReasoningEngine(api_key="", model="deepseek-v4-flash")

    assert engine.id() != "neurosymbolic"
    with pytest.raises(ValueError, match="DeepSeek engine is not configured"):
        engine.forward(make_prepared_argument())


def test_deepseek_prepare_renders_neurosymbolic_prompt_template():
    engine = make_engine()
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


def test_neurosymbolic_self_prompt_template_uses_plain_output_examples():
    prompt = prompt_registry.render("chat.self_prompt")

    assert '{"system": "<new system prompt>", "user": "<new user prompt>"}' in prompt
    assert '{"developer": "<new developer prompt>", "user": "<new user prompt>"}' in prompt
    assert "<system_output>" not in prompt
    assert "<developer_output>" not in prompt
    assert "<variant>" not in prompt
    assert "<requirement>" not in prompt


def test_engine_self_prompt_sends_prompt_object_as_raw_json():
    engine = make_engine(client_max_retries=0)
    captured = {}

    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        captured["body"] = body
        return httpx.Response(
            200,
            json=deepseek_response_json(
                content=json.dumps({"system": "new system prompt", "user": "new user prompt"}),
                reasoning_content=None,
            ),
            request=request,
        )

    repository = EngineRepository()
    previous_engine = repository._engines.get("neurosymbolic")
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            engine.transport_client = client
            EngineRepository.register("neurosymbolic", engine, allow_engine_override=True)
            result = engine.self_prompt({"system": "old system", "user": "old user"})
    finally:
        if previous_engine is not None:
            EngineRepository.register("neurosymbolic", previous_engine, allow_engine_override=True)
        else:
            repository._engines.pop("neurosymbolic", None)

    messages = captured["body"]["messages"]
    assert result == {"system": "new system prompt", "user": "new user prompt"}
    assert messages[0]["role"] == "system"
    assert "Generate a new system or developer prompt" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert json.loads(messages[1]["content"]) == {"system": "old system", "user": "old user"}
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_deepseek_build_request_returns_wire_request_and_safe_body():
    engine = make_engine(client_timeout=7.0)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function"}],
        },
        {"role": "user", "content": "hello"},
    ]
    argument = make_prepared_argument(
        messages=messages,
        kwargs={
            "temperature": 0.2,
            "max_tokens": 32,
            "extra_headers": {"X-Test": "1"},
            "extra_query": {"debug": "1"},
            "extra_body": {"temperature": 9, "vendor_flag": True},
        },
    )

    request = engine.build_request(argument)
    body = request.body()

    assert request.provider == "deepseek"
    assert request.operation == "chat.completions.create"
    assert request.method == "POST"
    assert request.url == DEEPSEEK_CHAT_COMPLETIONS_URL
    assert request.headers["Authorization"] == f"Bearer {DUMMY_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["X-Test"] == "1"
    assert request.params == {"debug": "1"}
    assert request.timeout == 7.0

    assert body["model"] == "deepseek-v4-flash"
    assert body["messages"][0]["content"] == ""
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 32
    assert body["stop"] == "<|endoftext|>"
    assert body["vendor_flag"] is True
    assert "extra_body" not in body
    assert "extra_headers" not in body
    assert "extra_query" not in body


def test_deepseek_build_request_roundtrips_null_content_tool_call_messages():
    engine = make_engine()
    messages = [
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "2"},
    ]

    request = engine.build_request(make_prepared_argument(messages=messages))
    body_messages = request.body()["messages"]

    assert "content" not in body_messages[1]
    assert body_messages[1]["tool_calls"] == [{"id": "call_1", "type": "function"}]
    assert body_messages[2]["content"] == "2"

    with pytest.raises(ValidationError):
        engine.build_request(make_prepared_argument(messages=[{"role": "user", "content": 1}]))


def test_deepseek_build_request_omits_max_tokens_when_not_provided():
    engine = make_engine()
    request = engine.build_request(make_prepared_argument())

    assert "max_tokens" not in request.body()


def test_deepseek_build_request_timeout_prefers_kwarg_then_client_timeout():
    engine = make_engine(client_timeout=7.0)
    bare_engine = make_engine()

    default_request = engine.build_request(make_prepared_argument())
    explicit_request = engine.build_request(make_prepared_argument(kwargs={"timeout": 11.0}))
    bare_request = bare_engine.build_request(make_prepared_argument())

    assert default_request.timeout == 7.0
    assert explicit_request.timeout == 11.0
    assert bare_request.timeout is None


def test_deepseek_build_request_rejects_bad_provider_kwargs():
    engine = make_engine()

    with pytest.raises(ValidationError):
        engine.build_request(make_prepared_argument(kwargs={"max_tokens": "32"}))

    with pytest.raises(ValueError, match="Unsupported request kwargs"):
        engine.build_request(
            make_prepared_argument(kwargs={"strict_request_kwargs": True, "temprature": 0.2})
        )


def test_deepseek_forward_uses_http_transport_and_typed_response():
    engine = make_engine(client_max_retries=0)
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=deepseek_response_json(), request=request)

    argument = Argument(
        args=("What is 1+1?",),
        signature_kwargs={"max_tokens": 16, "extra_query": {"debug": "1"}},
        decorator_kwargs={"prompt": "Answer the question.", "examples": []},
    )
    argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
    argument.prop.processed_input = "What is 1+1?"

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine.transport_client = client
        engine.prepare(argument)
        output, metadata = engine.forward(argument)

    assert [message["role"] for message in argument.prop.prepared_input] == ["system", "user"]
    assert captured["method"] == "POST"
    assert captured["url"] == f"{DEEPSEEK_CHAT_COMPLETIONS_URL}?debug=1"
    assert captured["authorization"] == f"Bearer {DUMMY_KEY}"
    assert captured["body"]["model"] == "deepseek-v4-flash"
    assert captured["body"]["max_tokens"] == 16
    assert captured["body"]["messages"] == argument.prop.prepared_input
    assert output == ["2"]
    assert metadata["thinking"] == "Add one and one."
    assert isinstance(metadata["raw_output"], DeepSeekResponse)


def test_deepseek_call_request_fails_fast_when_response_shape_drops_content():
    engine = make_engine(client_max_retries=0)

    def handler(request):
        payload = deepseek_response_json()
        del payload["choices"][0]["message"]["content"]
        return httpx.Response(200, json=payload, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine.transport_client = client
        request = engine.build_request(make_prepared_argument(kwargs={"max_tokens": 16}))
        with pytest.raises(ValidationError):
            engine.call_request(request)


def test_deepseek_call_request_fails_fast_when_response_drops_usage():
    engine = make_engine(client_max_retries=0)

    def handler(request):
        payload = deepseek_response_json()
        del payload["usage"]
        return httpx.Response(200, json=payload, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine.transport_client = client
        request = engine.build_request(make_prepared_argument(kwargs={"max_tokens": 16}))
        with pytest.raises(ValidationError):
            engine.call_request(request)


def test_deepseek_metadata_tracker_accumulates_usage():
    engine = make_engine(client_max_retries=0)

    def handler(request):
        return httpx.Response(200, json=deepseek_response_json(), request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine.transport_client = client
        with MetadataTracker() as tracker:
            engine.forward(make_prepared_argument(kwargs={"max_tokens": 16}))
            engine.forward(make_prepared_argument(kwargs={"max_tokens": 16}))
        usage = tracker.usage

    details = usage[("DeepSeekXReasoningEngine", "deepseek-v4-flash")]
    assert details["usage"]["prompt_tokens"] == 20
    assert details["usage"]["completion_tokens"] == 10
    assert details["usage"]["total_tokens"] == 30
    assert details["usage"]["total_calls"] == 2
    assert details["completion_breakdown"]["reasoning_tokens"] == 6
    assert "thinking_content" not in details


def test_deepseek_forward_streams_sse_and_aggregates_response():
    engine = make_engine(client_max_retries=0)
    captured = {}
    chunks = [
        {
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "reasoning_content": "Add one "}}
            ]
        },
        {"choices": [{"index": 0, "delta": {"reasoning_content": "and one."}}]},
        {"choices": [{"index": 0, "delta": {"content": "2"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]

    def handler(request):
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content=sse_stream_body(chunks),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine.transport_client = client
        output, metadata = engine.forward(make_prepared_argument(kwargs={"stream": True}))

    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert output == ["2"]
    assert metadata["thinking"] == "Add one and one."
    raw_output = metadata["raw_output"]
    assert isinstance(raw_output, DeepSeekResponse)
    assert raw_output.choices[0]["finish_reason"] == "stop"
    assert raw_output.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_deepseek_stream_fails_fast_when_usage_chunk_is_missing():
    engine = make_engine(client_max_retries=0)
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "2"}, "finish_reason": "stop"}]},
    ]

    def handler(request):
        return httpx.Response(
            200,
            content=sse_stream_body(chunks),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine.transport_client = client
        argument = make_prepared_argument(
            kwargs={"stream": True, "stream_options": {"include_usage": False}}
        )
        with pytest.raises(ValidationError):
            engine.forward(argument)


@pytest.mark.engine_live
@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_deepseek_live_smoke(engine_api_mode, model):
    if engine_api_mode != "live":
        pytest.skip("use --engine-api=live to run live engine API requests")

    api_key = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_API_KEY", "")
    if not api_key:
        pytest.skip("symai.config.json NEUROSYMBOLIC_ENGINE_API_KEY is required")

    engine = DeepSeekXReasoningEngine(
        api_key=api_key,
        model=model,
        client_timeout=30.0,
        client_max_retries=0,
    )
    argument = Argument(
        args=("Reply with exactly: ok",),
        signature_kwargs={"max_tokens": 16},
        decorator_kwargs={"prompt": "Follow the user instruction exactly.", "examples": []},
    )
    argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
    argument.prop.processed_input = "Reply with exactly: ok"

    engine.prepare(argument)
    output, metadata = engine.forward(argument)

    assert isinstance(output[0], str)
    assert isinstance(metadata["raw_output"], DeepSeekResponse)


@pytest.mark.engine_live
def test_deepseek_live_stream_smoke(engine_api_mode):
    if engine_api_mode != "live":
        pytest.skip("use --engine-api=live to run live engine API requests")

    api_key = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_API_KEY", "")
    if not api_key:
        pytest.skip("symai.config.json NEUROSYMBOLIC_ENGINE_API_KEY is required")

    engine = DeepSeekXReasoningEngine(
        api_key=api_key,
        model="deepseek-v4-flash",
        client_timeout=30.0,
        client_max_retries=0,
    )
    argument = Argument(
        args=("Reply with exactly: ok",),
        signature_kwargs={"max_tokens": 16, "stream": True},
        decorator_kwargs={"prompt": "Follow the user instruction exactly.", "examples": []},
    )
    argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
    argument.prop.processed_input = "Reply with exactly: ok"

    engine.prepare(argument)
    output, metadata = engine.forward(argument)

    assert isinstance(output[0], str)
    raw_output = metadata["raw_output"]
    assert isinstance(raw_output, DeepSeekResponse)
    assert raw_output.usage["total_tokens"] > 0
