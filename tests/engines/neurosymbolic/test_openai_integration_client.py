from types import SimpleNamespace

import httpx

from symai.backend.engines.neurosymbolic.engine_openai_gptX import OpenAIResponsesEngine
from symai.backend.integrations.openai.responses import ResponsesResponse

DUMMY_KEY = "sk-test-not-a-real-key"


def _argument(**kwargs):
    return SimpleNamespace(
        kwargs=kwargs,
        prop=SimpleNamespace(
            prepared_input=[{"role": "user", "content": "hello"}],
        ),
    )


def test_engine_executes_through_standalone_openai_client():
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            headers={"x-request-id": "request-id"},
            json={
                "id": "response-id",
                "object": "response",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "thought"}],
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "answer"}],
                    },
                    {
                        "type": "function_call",
                        "name": "lookup",
                        "arguments": '{"query":"x"}',
                        "call_id": "call-1",
                    },
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        engine = OpenAIResponsesEngine(
            api_key=DUMMY_KEY,
            model="openai:gpt-5.4",
            http_client=http_client,
        )
        output, metadata = engine.forward(_argument(max_output_tokens=16))

    assert output == ["answer"]
    assert metadata["thinking"] == "thought"
    assert metadata["function_call"] == {
        "name": "lookup",
        "arguments": {"query": "x"},
        "call_id": "call-1",
    }
    assert isinstance(metadata["raw_output"], ResponsesResponse)
    assert metadata["response"].metadata.request_id == "request-id"
