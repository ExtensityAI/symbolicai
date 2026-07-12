import httpx
import pytest

from symai.backend.integrations.openai.client import Client
from symai.backend.integrations.openai.embeddings import EmbeddingRequest
from symai.backend.integrations.openai.errors import AuthError, ResponseError
from symai.backend.integrations.openai.responses import ResponsesRequest


def test_responses_posts_full_request_and_parses_output():
    def handler(request: httpx.Request):
        assert request.method == "POST"
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.read() == (
            b'{"input":[{"role":"user","content":"hello"}],'
            b'"model":"gpt-5.4","max_output_tokens":32,'
            b'"reasoning":{"effort":"medium"},"tools":'
            b'[{"type":"web_search"}]}'
        )
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
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        )

    request = ResponsesRequest.model_validate(
        {
            "input": ({"role": "user", "content": "hello"},),
            "model": "gpt-5.4",
            "max_output_tokens": 32,
            "reasoning": {"effort": "medium"},
            "tools": ({"type": "web_search"},),
        }
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        response = Client(api_key="test-key", http_client=http_client).responses(request)

    assert response.data.output[1].content[0].text == "answer"
    assert response.data.usage.total_tokens == 3
    assert response.metadata.request_id == "request-id"


def test_embeddings_posts_batch_and_parses_vectors():
    def handler(request: httpx.Request):
        assert str(request.url) == "https://api.openai.com/v1/embeddings"
        assert request.read() == (
            b'{"input":["one","two"],"model":"text-embedding-3-small",'
            b'"dimensions":2,"encoding_format":"float"}'
        )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [1.0, 0.0], "index": 0},
                    {"object": "embedding", "embedding": [0.0, 1.0], "index": 1},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    request = EmbeddingRequest(
        input=("one", "two"),
        model="text-embedding-3-small",
        dimensions=2,
        encoding_format="float",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        response = Client(api_key="test-key", http_client=http_client).embeddings(request)

    assert response.data.data[0].embedding == (1.0, 0.0)
    assert response.data.usage.prompt_tokens == 2


def test_client_maps_auth_and_invalid_success_responses():
    responses = iter(
        [
            httpx.Response(401, text="unauthorized"),
            httpx.Response(200, json={"output": "wrong"}),
        ]
    )

    def handler(_request: httpx.Request):
        return next(responses)

    request = ResponsesRequest(input="hello", model="gpt-5.4")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = Client(api_key="test-key", http_client=http_client)
        with pytest.raises(AuthError):
            client.responses(request)
        with pytest.raises(ResponseError):
            client.responses(request)
