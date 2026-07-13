import pytest
from pydantic import ValidationError

from symai.clients.cerebras import chat as cerebras_chat
from symai.clients.deepseek import chat as deepseek_chat
from symai.clients.openai import embeddings as openai_embeddings
from symai.clients.openai import responses as openai_responses


def test_endpoint_packages_own_model_catalogs():
    openai_reasoning = openai_responses.MODEL_SPECS["gpt-5.4"].reasoning
    cerebras_reasoning = cerebras_chat.MODEL_SPECS["gpt-oss-120b"].reasoning
    deepseek_reasoning = deepseek_chat.MODEL_SPECS["deepseek-v4-flash"].reasoning

    assert openai_reasoning is not None
    assert cerebras_reasoning is not None
    assert deepseek_reasoning is not None
    assert openai_embeddings.MODEL_SPECS["text-embedding-3-large"].dimensions == 3_072
    assert cerebras_chat.MODEL_SPECS["gpt-oss-120b"].context_tokens == 131_072
    assert deepseek_chat.MODEL_SPECS["deepseek-v4-flash"].vision is False


def test_chat_requests_accept_nonempty_future_model_ids():
    deepseek_request = deepseek_chat.CreateChatCompletionRequest(
        messages=(deepseek_chat.UserMessage(role="user", content="hello"),),
        model="future-deepseek-model",
    )
    cerebras_request = cerebras_chat.CreateChatCompletionRequest(
        messages=(cerebras_chat.UserMessage(role="user", content="hello"),),
        model="future-cerebras-model",
    )

    assert deepseek_request.model == "future-deepseek-model"
    assert cerebras_request.model == "future-cerebras-model"
    with pytest.raises(ValidationError):
        deepseek_chat.CreateChatCompletionRequest(
            messages=(deepseek_chat.UserMessage(role="user", content="hello"),),
            model="",
        )
    with pytest.raises(ValidationError):
        cerebras_chat.CreateChatCompletionRequest(
            messages=(cerebras_chat.UserMessage(role="user", content="hello"),),
            model="",
        )
