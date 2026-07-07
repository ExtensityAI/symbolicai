import json

import httpx

from symai.backend.providers.cerebras.client import CerebrasClient, extract_thinking
from symai.backend.providers.cerebras.request import (
    CerebrasResponseFormat,
    ChatRequest,
    JsonSchemaSpec,
    Message,
    Role,
)
from symai.backend.providers.cerebras.response import ChatResponse
from symai.backend.providers.cerebras.spec import CerebrasModel


def _chat_request() -> ChatRequest:
    schema_spec = JsonSchemaSpec(name="Answer", json_schema_body={"type": "object"})
    response_format = CerebrasResponseFormat(type="json_schema", json_schema=schema_spec)
    return ChatRequest(
        messages=(Message(role=Role.USER, content="hi"),),
        model=CerebrasModel.GPT_OSS_120B,
        response_format=response_format,
    )


def _completion_json() -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# --- extract_thinking ---------------------------------------------------------


def test_extract_thinking_returns_trimmed_thinking_and_content():
    thinking, content = extract_thinking("<think>reasoning here</think>the answer")

    assert thinking == "reasoning here"
    assert content == "the answer"


def test_extract_thinking_captures_multiline_block_with_dotall():
    raw = "<think>line one\nline two\nline three</think>final answer"

    thinking, content = extract_thinking(raw)

    assert thinking == "line one\nline two\nline three"
    assert content == "final answer"


def test_extract_thinking_no_tags_returns_none_and_same_content():
    raw = "just a plain answer, no reasoning tags here"

    thinking, content = extract_thinking(raw)

    assert thinking is None
    assert content == raw


def test_extract_thinking_empty_block_returns_none_thinking_and_stripped_content():
    thinking, content = extract_thinking("<think></think>the answer")

    assert thinking is None
    assert content == "the answer"


# --- create() happy path -------------------------------------------------------


def test_create_success_posts_expected_request_and_parses_response():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_completion_json(), request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = CerebrasClient(api_key="test-key", http_client=http_client)

    response = client.create(_chat_request())

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/chat/completions")
    assert captured["authorization"] == "Bearer test-key"
    assert "model" in captured["body"]
    assert captured["body"]["messages"]
    assert captured["body"]["response_format"]["json_schema"]["schema"] == {"type": "object"}

    assert isinstance(response, ChatResponse)
    assert response.choices[0].message.content == "hello there"
    assert response.usage.total_tokens == 15
