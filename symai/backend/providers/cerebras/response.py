from symai.backend.providers.models import TolerantModel


class PromptTokensDetails(TolerantModel):
    cached_tokens: int | None = None


class CompletionTokensDetails(TolerantModel):
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None
    reasoning_tokens: int | None = None


class Usage(TolerantModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    image_tokens: int | None = None
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


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
