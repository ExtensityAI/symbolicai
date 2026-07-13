import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.chat_prompts import prompt_registry
from symai.backend.engines.language_model.deepseek import LanguageModelEngine
from symai.backend.settings import SYMAI_CONFIG
from symai.clients.deepseek.chat import MODEL_SPECS, ChatCompletion
from symai.clients.deepseek.client import Client
from symai.clients.deepseek.errors import ResponseError
from symai.components import MetadataTracker
from symai.context import CURRENT_ENGINE_VAR
from symai.core import Argument

DUMMY_KEY = "sk-test-not-a-real-key"


def make_engine(http_client, model="deepseek-v4-flash", api_key=DUMMY_KEY):
    return LanguageModelEngine(
        client=Client(api_key=api_key, http_client=http_client),
        model=model,
    )


@pytest.fixture
def http_client():
    with httpx.Client() as client:
        yield client


def make_prepared_argument(kwargs=None, messages=None):
    if kwargs is None:
        kwargs = {}
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return SimpleNamespace(kwargs=kwargs, prop=SimpleNamespace(prepared_input=messages))


def deepseek_response_json(content="2", reasoning_content="Add one and one."):
    return {
        "id": "chatcmpl-test",
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
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


def test_deepseek_supported_models_track_client_capabilities():
    assert tuple(MODEL_SPECS) == ("deepseek-v4-flash", "deepseek-v4-pro")
    for spec in MODEL_SPECS.values():
        assert spec.reasoning is not None
        assert spec.vision is False


def test_deepseek_prepare_renders_neurosymbolic_prompt_template(http_client):
    engine = make_engine(http_client)
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

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine = make_engine(client)
        token = CURRENT_ENGINE_VAR.set(engine)
        try:
            result = engine.self_prompt({"system": "old system", "user": "old user"})
        finally:
            CURRENT_ENGINE_VAR.reset(token)

    messages = captured["body"]["messages"]
    assert result == {"system": "new system prompt", "user": "new user prompt"}
    assert messages[0]["role"] == "system"
    assert "Generate a new system or developer prompt" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert json.loads(messages[1]["content"]) == {"system": "old system", "user": "old user"}
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_deepseek_build_request_returns_standalone_client_request(http_client):
    engine = make_engine(http_client)
    argument = make_prepared_argument(
        kwargs={
            "temperature": 0.2,
            "max_tokens": 32,
            "thinking": {"type": "enabled"},
        },
    )

    request = engine.build_request(argument)
    body = request.model_dump(mode="json", exclude_none=True)

    assert body == {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "deepseek-v4-flash",
        "thinking": {"type": "enabled"},
        "max_tokens": 32,
        "temperature": 0.2,
    }


def test_deepseek_build_request_rejects_tool_messages(http_client):
    engine = make_engine(http_client)
    messages = [
        {"role": "user", "content": "What is 1+1?"},
        {"role": "tool", "tool_call_id": "call_1", "content": "2"},
    ]

    with pytest.raises(ValidationError):
        engine.build_request(make_prepared_argument(messages=messages))

    with pytest.raises(ValidationError):
        engine.build_request(make_prepared_argument(messages=[{"role": "user", "content": 1}]))


def test_deepseek_build_request_omits_max_tokens_when_not_provided(http_client):
    engine = make_engine(http_client)
    request = engine.build_request(make_prepared_argument())

    assert request.max_tokens is None


def test_deepseek_build_request_rejects_bad_provider_kwargs(http_client):
    engine = make_engine(http_client)

    with pytest.raises(ValidationError):
        engine.build_request(make_prepared_argument(kwargs={"max_tokens": "32"}))

    with pytest.raises(ValueError, match="Unsupported request kwargs"):
        engine.build_request(
            make_prepared_argument(kwargs={"strict_request_kwargs": True, "temprature": 0.2})
        )


def test_deepseek_forward_uses_http_transport_and_typed_response():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=deepseek_response_json(), request=request)

    argument = Argument(
        args=("What is 1+1?",),
        signature_kwargs={"max_tokens": 16},
        decorator_kwargs={"prompt": "Answer the question.", "examples": []},
    )
    argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
    argument.prop.processed_input = "What is 1+1?"

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine = make_engine(client)
        engine.prepare(argument)
        output, metadata = engine.forward(argument)

    assert [message["role"] for message in argument.prop.prepared_input] == ["system", "user"]
    assert captured["method"] == "POST"
    assert captured["authorization"] == f"Bearer {DUMMY_KEY}"
    assert captured["body"]["model"] == "deepseek-v4-flash"
    assert captured["body"]["max_tokens"] == 16
    assert captured["body"]["messages"] == argument.prop.prepared_input
    assert output == ["2"]
    assert metadata["thinking"] == "Add one and one."
    assert isinstance(metadata["raw_output"], ChatCompletion)


def test_deepseek_call_request_fails_fast_when_response_shape_drops_content():

    def handler(request):
        payload = deepseek_response_json()
        del payload["choices"][0]["message"]["content"]
        return httpx.Response(200, json=payload, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine = make_engine(client)
        request = engine.build_request(make_prepared_argument(kwargs={"max_tokens": 16}))
        with pytest.raises(ResponseError):
            engine.call_request(request)


def test_deepseek_call_request_fails_fast_when_response_drops_usage():

    def handler(request):
        payload = deepseek_response_json()
        del payload["usage"]
        return httpx.Response(200, json=payload, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine = make_engine(client)
        request = engine.build_request(make_prepared_argument(kwargs={"max_tokens": 16}))
        with pytest.raises(ResponseError):
            engine.call_request(request)


def test_deepseek_metadata_tracker_accumulates_usage():

    def handler(request):
        return httpx.Response(200, json=deepseek_response_json(), request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine = make_engine(client)
        with MetadataTracker() as tracker:
            engine.forward(make_prepared_argument(kwargs={"max_tokens": 16}))
            engine.forward(make_prepared_argument(kwargs={"max_tokens": 16}))
        usage = tracker.usage

    details = usage[("deepseek.language_model", "deepseek-v4-flash")]
    assert details["usage"]["prompt_tokens"] == 20
    assert details["usage"]["completion_tokens"] == 10
    assert details["usage"]["total_tokens"] == 30
    assert details["usage"]["total_calls"] == 2
    assert details["completion_breakdown"]["reasoning_tokens"] == 6
    assert "thinking_content" not in details


@pytest.mark.engine_live
@pytest.mark.parametrize("model", MODEL_SPECS)
def test_deepseek_live_smoke(engine_api_mode, model):
    if engine_api_mode != "live":
        pytest.skip("use --engine-api=live to run live engine API requests")

    api_key = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_API_KEY", "")
    if not api_key:
        pytest.skip("symai.config.json NEUROSYMBOLIC_ENGINE_API_KEY is required")

    argument = Argument(
        args=("Reply with exactly: ok",),
        signature_kwargs={"max_tokens": 16},
        decorator_kwargs={"prompt": "Follow the user instruction exactly.", "examples": []},
    )
    argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
    argument.prop.processed_input = "Reply with exactly: ok"

    with httpx.Client(timeout=30.0) as client:
        engine = make_engine(client, model=model, api_key=api_key)
        engine.prepare(argument)
        output, metadata = engine.forward(argument)

    assert isinstance(output[0], str)
    assert isinstance(metadata["raw_output"], ChatCompletion)
