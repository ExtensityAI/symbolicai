from symai.clients import cerebras, deepseek, openai


def test_openai_facade_exposes_client_endpoints_and_errors():
    assert openai.Client.__module__ == "symai.clients.openai.client"
    request = openai.embeddings.CreateEmbeddingRequest(
        input="hello",
        model="text-embedding-3-small",
    )
    assert request.model == "text-embedding-3-small"
    assert issubclass(openai.errors.APIError, Exception)
    assert openai.responses.MODEL_SPECS


def test_cerebras_facade_exposes_client_endpoint_and_errors():
    assert cerebras.Client.__module__ == "symai.clients.cerebras.client"
    request = cerebras.chat.CreateChatCompletionRequest(
        messages=(cerebras.chat.UserMessage(role="user", content="hello"),),
        model="gpt-oss-120b",
    )
    assert request.model == "gpt-oss-120b"
    assert issubclass(cerebras.errors.APIError, Exception)


def test_deepseek_facade_exposes_client_endpoint_and_errors():
    assert deepseek.Client.__module__ == "symai.clients.deepseek.client"
    request = deepseek.chat.CreateChatCompletionRequest(
        messages=(deepseek.chat.UserMessage(role="user", content="hello"),),
        model="deepseek-v4-flash",
    )
    assert request.model == "deepseek-v4-flash"
    assert issubclass(deepseek.errors.APIError, Exception)
