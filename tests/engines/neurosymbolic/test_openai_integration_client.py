from types import SimpleNamespace

import httpx
import pytest

from symai.backend.engines.language_model.openai import LanguageModelEngine
from symai.clients.openai.client import Client
from symai.clients.openai.responses import Response

DUMMY_KEY = "sk-test-not-a-real-key"


def _argument(**kwargs):
    return SimpleNamespace(
        kwargs=kwargs,
        prop=SimpleNamespace(
            prepared_input=[{"role": "user", "content": "hello"}],
        ),
    )


def _engine(http_client):
    return LanguageModelEngine(
        client=Client(api_key=DUMMY_KEY, http_client=http_client),
        model="gpt-5.4",
    )


def test_engine_executes_through_comprehensive_responses_client():
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            headers={"x-request-id": "request-id"},
            json={
                "id": "response-id",
                "object": "response",
                "created_at": 1.5,
                "status": "completed",
                "background": False,
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "max_output_tokens": 16,
                "model": "gpt-5.4",
                "output": [
                    {
                        "id": "reasoning-id",
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [{"type": "summary_text", "text": "thought"}],
                    },
                    {
                        "id": "message-id",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "answer",
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    },
                ],
                "store": True,
                "truncation": "disabled",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
                "metadata": {},
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        engine = _engine(http_client)
        output, metadata = engine.forward(_argument(max_output_tokens=16))

    assert output == ["answer"]
    assert metadata["thinking"] == "thought"
    assert "function_call" not in metadata
    assert isinstance(metadata["raw_output"], Response)
    assert metadata["response"].metadata.request_id == "request-id"


def test_engine_rejects_background_requests():
    with httpx.Client() as http_client:
        engine = _engine(http_client)

    with pytest.raises(ValueError, match="background"):
        engine.build_request(_argument(background=True))


@pytest.mark.parametrize("status", ["queued", "in_progress", "failed"])
def test_engine_rejects_non_completed_responses(status):
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "id": "response-id",
                "object": "response",
                "created_at": 1.5,
                "status": status,
                "background": status != "failed",
                "error": (
                    {"code": "server_error", "message": "generation failed"}
                    if status == "failed"
                    else None
                ),
                "incomplete_details": None,
                "instructions": None,
                "max_output_tokens": 16,
                "model": "gpt-5.4",
                "output": [],
                "store": True,
                "truncation": "disabled",
                "usage": None,
                "metadata": {},
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        engine = _engine(http_client)
        with pytest.raises(ValueError, match=status):
            engine.forward(_argument())
