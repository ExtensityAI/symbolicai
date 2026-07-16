from symai.providers.cerebras.client import chat as cerebras_chat
from symai.providers.cerebras.client import errors as cerebras_errors
from symai.providers.deepseek.client import chat as deepseek_chat
from symai.providers.deepseek.client import errors as deepseek_errors
from symai.providers.openai.client import embeddings as openai_embeddings
from symai.providers.openai.client import errors as openai_errors
from symai.providers.openai.client import responses as openai_responses


def test_openai_client_package_exposes_endpoints_and_errors():
    request = openai_embeddings.CreateEmbeddingRequest(
        input="hello",
        model="text-embedding-3-small",
    )
    assert request.model == "text-embedding-3-small"
    assert issubclass(openai_errors.APIError, Exception)
    assert openai_responses.MODEL_SPECS


def test_cerebras_client_package_exposes_endpoint_and_errors():
    request = cerebras_chat.CreateChatCompletionRequest(
        messages=(cerebras_chat.UserMessage(role="user", content="hello"),),
        model="gpt-oss-120b",
    )
    assert request.model == "gpt-oss-120b"
    assert issubclass(cerebras_errors.APIError, Exception)


def test_deepseek_client_package_exposes_endpoint_and_errors():
    request = deepseek_chat.CreateChatCompletionRequest(
        messages=(deepseek_chat.UserMessage(role="user", content="hello"),),
        model="deepseek-v4-flash",
    )
    assert request.model == "deepseek-v4-flash"
    assert issubclass(deepseek_errors.APIError, Exception)
