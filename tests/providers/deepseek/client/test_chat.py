import pytest
from pydantic import ValidationError

from symai.providers.deepseek.client.chat import (
    AssistantMessage,
    ChatCompletion,
    CreateChatCompletionRequest,
    JsonObjectResponseFormat,
    ReasoningEffort,
    SystemMessage,
    Thinking,
    ThinkingType,
    UserMessage,
)


def test_chat_request_serializes_supported_non_streaming_surface():
    request = CreateChatCompletionRequest(
        messages=(
            SystemMessage(role="system", content="Answer concisely"),
            UserMessage(role="user", content="Why?", name="caller"),
            AssistantMessage(role="assistant", content=None),
        ),
        model="deepseek-v4-pro",
        thinking=Thinking(type=ThinkingType.ENABLED),
        reasoning_effort=ReasoningEffort.MAX,
        max_tokens=1024,
        response_format=JsonObjectResponseFormat(type="json_object"),
        stop=("END",),
        temperature=0.5,
        top_p=0.9,
        user_id="customer_42",
    )

    assert request.model_dump(mode="json", exclude_none=True) == {
        "messages": [
            {"role": "system", "content": "Answer concisely"},
            {"role": "user", "content": "Why?", "name": "caller"},
            {"role": "assistant"},
        ],
        "model": "deepseek-v4-pro",
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "stop": ["END"],
        "temperature": 0.5,
        "top_p": 0.9,
        "user_id": "customer_42",
    }


def test_chat_request_accepts_new_model_ids_without_client_release():
    request = CreateChatCompletionRequest(
        messages=(UserMessage(role="user", content="hello"),),
        model="deepseek-future-model",
    )

    assert request.model == "deepseek-future-model"


def test_assistant_message_rejects_beta_prefix_fields():
    with pytest.raises(ValidationError):
        AssistantMessage.model_validate(
            {
                "role": "assistant",
                "content": "prefix",
                "prefix": True,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("messages", ()),
        ("max_tokens", 0),
        ("stop", tuple(str(index) for index in range(17))),
        ("temperature", 2.1),
        ("top_p", 1.1),
        ("user_id", "contains spaces"),
        ("user_id", "x" * 513),
    ],
)
def test_chat_request_rejects_values_outside_documented_bounds(field, value):
    payload = {
        "messages": (UserMessage(role="user", content="hello"),),
        "model": "deepseek-v4-flash",
        field: value,
    }

    with pytest.raises(ValidationError):
        CreateChatCompletionRequest(**payload)


def test_chat_request_rejects_streaming_tools_and_unknown_fields():
    payload = {
        "messages": (UserMessage(role="user", content="hello"),),
        "model": "deepseek-v4-flash",
        "stream": True,
        "tools": (),
    }

    with pytest.raises(ValidationError):
        CreateChatCompletionRequest(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logprobs", True),
        ("top_logprobs", 5),
        ("logit_bias", {"123": 1.0}),
    ],
)
def test_chat_request_rejects_removed_logprob_fields(field: str, value: object):
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "deepseek-v4-flash",
                "messages": (UserMessage(role="user", content="hello"),),
                field: value,
            }
        )


def test_chat_response_models_reasoning_cache_usage_and_logprobs():
    response = ChatCompletion.model_validate(
        {
            "id": "response-id",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "thought",
                    },
                    "logprobs": {
                        "content": [
                            {
                                "token": "answer",
                                "logprob": -0.1,
                                "bytes": [97],
                                "top_logprobs": [],
                            }
                        ],
                        "reasoning_content": None,
                    },
                }
            ],
            "created": 1,
            "model": "deepseek-v4-pro",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 4,
                "prompt_tokens": 3,
                "prompt_cache_hit_tokens": 2,
                "prompt_cache_miss_tokens": 1,
                "total_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
            "future_field": {"kept": True},
        }
    )

    choice = response.choices[0]
    completion_details = response.usage.completion_tokens_details
    logprobs = choice.logprobs

    assert completion_details is not None
    assert logprobs is not None
    assert logprobs.content is not None
    assert choice.message.reasoning_content == "thought"
    assert response.usage.prompt_cache_hit_tokens == 2
    assert completion_details.reasoning_tokens == 2
    assert logprobs.content[0].token == "answer"
    assert response.model_extra == {"future_field": {"kept": True}}
