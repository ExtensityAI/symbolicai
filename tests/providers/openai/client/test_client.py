import httpx
import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from symai.providers._client.headers import _UNSAFE_API_KEY_MESSAGE
from symai.providers.openai.client.client import Client
from symai.providers.openai.client.embeddings import CreateEmbeddingRequest, EmbeddingList
from symai.providers.openai.client.errors import AuthError, ResponseError
from symai.providers.openai.client.responses import (
    CreateResponseRequest,
    InputMessage,
    InputText,
    OutputMessage,
    OutputText,
    ReasoningConfig,
    ReasoningEffort,
    ReasoningSummary,
    Response,
    URLCitation,
)


@pytest.mark.parametrize(
    "api_key",
    [
        "",
        "secret\rvalue",
        "secret\nvalue",
        "secret\x01value",
        "secret\x7fvalue",
        " secret",
        "secret ",
        "api_key ",
        " ",
        "ValueError\n",
        "TypeError\n",
    ],
    ids=[
        "empty",
        "cr",
        "lf",
        "c0",
        "del",
        "leading-space",
        "trailing-space",
        "message-collision",
        "whitespace-only",
        "value-error-traceback-text",
        "type-error-traceback-text",
    ],
)
def test_client_rejects_unsafe_api_key_before_request_without_disclosure(
    api_key: str,
):
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200)

    with pytest.raises(ValueError) as exc_info:
        Client(
            api_key=SecretStr(api_key),
            transport=httpx.MockTransport(handler),
        )

    assert attempts == 0
    # A constant message cannot be derived from the credential, and every invalid
    # credential must fail identically: which rule a key trips is a property of the secret.
    assert exc_info.value.args == (_UNSAFE_API_KEY_MESSAGE,)


def test_client_rejects_plaintext_api_key():
    with pytest.raises(TypeError) as exc_info:
        Client(
            api_key="test-key",  # pyright: ignore[reportArgumentType]
            transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
        )

    assert exc_info.value.args == ("api_key must be a SecretStr",)


def _minimal_response_json(status: str = "completed"):
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 1.5,
        "status": status,
        "background": status != "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-5.4",
        "output": [],
        "store": True,
        "truncation": "disabled",
        "usage": None,
        "metadata": {},
    }


def test_responses_posts_typed_request_without_tool_surface():
    def handler(request: httpx.Request):
        assert request.method == "POST"
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.read() == (
            b'{"input":[{"role":"user","content":'
            b'[{"type":"input_text","text":"hello"}],"type":"message"}],'
            b'"model":"gpt-5.4",'
            b'"max_output_tokens":32,"reasoning":{"effort":"medium",'
            b'"summary":"detailed"},"store":false}'
        )
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
                "instructions": "Be concise",
                "max_output_tokens": 32,
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
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.com",
                                        "title": "Example",
                                        "start_index": 0,
                                        "end_index": 6,
                                    }
                                ],
                                "logprobs": [],
                            }
                        ],
                    },
                ],
                "previous_response_id": None,
                "reasoning": {"effort": "medium", "summary": "detailed"},
                "service_tier": "default",
                "store": False,
                "temperature": 1.0,
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "top_p": 1.0,
                "truncation": "disabled",
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 2,
                    "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": 3,
                },
                "user": None,
                "metadata": {"request": "test"},
            },
        )

    request = CreateResponseRequest(
        input=(
            InputMessage(
                role="user",
                content=(InputText(type="input_text", text="hello"),),
            ),
        ),
        model="gpt-5.4",
        max_output_tokens=32,
        reasoning=ReasoningConfig(
            effort=ReasoningEffort.MEDIUM,
            summary=ReasoningSummary.DETAILED,
        ),
        store=False,
    )
    client = Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )
    try:
        response = client.create_response(request)
    finally:
        client.close()

    output = response.data.output[1]
    assert isinstance(output, OutputMessage)
    content = output.content[0]
    assert isinstance(content, OutputText)
    assert content.text == "answer"
    annotation = content.annotations[0]
    assert isinstance(annotation, URLCitation)
    assert annotation.url == "https://example.com"
    assert response.data.usage is not None
    assert response.data.usage.output_tokens_details.reasoning_tokens == 1
    assert response.metadata.request_id == "request-id"
    assert isinstance(response, BaseModel)
    assert isinstance(response.metadata, BaseModel)


def test_responses_request_rejects_tool_calling_parameters():
    with pytest.raises(ValidationError):
        CreateResponseRequest.model_validate(
            {
                "input": "hello",
                "model": "gpt-5.4",
                "tools": (),
                "tool_choice": "auto",
                "max_tool_calls": 1,
                "parallel_tool_calls": True,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logprobs", True),
        ("top_logprobs", 5),
        ("logit_bias", {"123": 1.0}),
    ],
)
def test_responses_request_rejects_removed_logprob_fields(field: str, value: object):
    with pytest.raises(ValidationError):
        CreateResponseRequest.model_validate(
            {
                "input": "hello",
                "model": "gpt-5.4",
                field: value,
            }
        )


def test_request_models_accept_nonempty_future_model_ids():
    response_request = CreateResponseRequest(
        input="hello",
        model="future-chat-model",
    )
    embedding_request = CreateEmbeddingRequest(
        input="hello",
        model="future-embedding-model",
    )

    assert response_request.model == "future-chat-model"
    assert embedding_request.model == "future-embedding-model"
    with pytest.raises(ValidationError):
        CreateResponseRequest(input="hello", model="")
    with pytest.raises(ValidationError):
        CreateEmbeddingRequest(input="hello", model="")


def test_response_models_accept_future_model_ids():
    response_payload = _minimal_response_json()
    response_payload["model"] = "future-response-model"
    embedding_payload = {
        "object": "list",
        "data": [],
        "model": "future-embedding-model",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }

    assert Response.model_validate(response_payload).model == "future-response-model"
    assert EmbeddingList.model_validate(embedding_payload).model == "future-embedding-model"


def test_response_client_exposes_only_used_endpoints():
    assert not hasattr(Client, "retrieve_response")
    assert not hasattr(Client, "delete_response")
    assert not hasattr(Client, "cancel_response")
    assert not hasattr(Client, "list_input_items")


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

    request = CreateEmbeddingRequest(
        input=("one", "two"),
        model="text-embedding-3-small",
        dimensions=2,
        encoding_format="float",
    )
    client = Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )
    try:
        response = client.create_embeddings(request)
    finally:
        client.close()

    assert response.data.data[0].embedding == (1.0, 0.0)
    assert response.data.usage.prompt_tokens == 2
    assert isinstance(response.data, BaseModel)


def test_client_maps_auth_and_invalid_success_responses():
    api_responses = iter(
        [
            httpx.Response(401, text="unauthorized"),
            httpx.Response(200, json={"output": "wrong"}),
        ]
    )

    def handler(_request: httpx.Request):
        return next(api_responses)

    request = CreateResponseRequest(input="hello", model="gpt-5.4")
    client = Client(
        api_key=SecretStr("test-key"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(AuthError):
            client.create_response(request)
        with pytest.raises(ResponseError):
            client.create_response(request)
    finally:
        client.close()
