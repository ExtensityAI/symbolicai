from typing import Any

from pydantic import model_validator

from symai.backend.providers.cerebras.base import TolerantModel


class Usage(TolerantModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _lift_reasoning_tokens(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        details = data.get("completion_tokens_details")
        if not isinstance(details, dict) or "reasoning_tokens" not in details:
            return data

        return {**data, "reasoning_tokens": details["reasoning_tokens"]}


class ResponseMessage(TolerantModel):
    role: str
    content: str | None = None
    reasoning: str | None = None


class Choice(TolerantModel):
    index: int
    message: ResponseMessage
    finish_reason: str | None = None


class ChatResponse(TolerantModel):
    choices: tuple[Choice, ...]
    usage: Usage
