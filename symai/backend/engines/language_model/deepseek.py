from typing import Any

from symai.backend.base import Engine
from symai.backend.chat_prompts import render_chat_system_prompt
from symai.backend.usage import EngineUsageRecord
from symai.clients.deepseek import chat as chat_api
from symai.clients.deepseek.chat import (
    ChatCompletion,
    CreateChatCompletionRequest,
    ReasoningEffort,
    Thinking,
)
from symai.clients.deepseek.client import Client as DeepSeekClient
from symai.clients.deepseek.transport import APIResponse


class LanguageModelEngine(Engine):
    provider = "deepseek"
    capability = "language_model"

    def __init__(self, *, client: DeepSeekClient, model: chat_api.Model):
        super().__init__()
        try:
            self.model_spec = chat_api.MODEL_SPECS[model]
        except KeyError as e:
            msg = f"Unsupported model: {model}"
            raise ValueError(msg) from e

        self.client = client
        self.model = model
        self.name = self.__class__.__name__
        self.tokenizer = None
        self.max_context_tokens = self.model_spec.context_tokens

    def compute_required_tokens(self, _messages: list[dict[str, Any]]) -> int:
        msg = 'Method "compute_required_tokens" not implemented for LanguageModelEngine.'
        raise NotImplementedError(msg)

    def compute_remaining_tokens(self, _prompts: list[dict[str, Any]]) -> int:
        msg = 'Method "compute_remaining_tokens" not implemented for LanguageModelEngine.'
        raise NotImplementedError(msg)

    def build_request(self, argument) -> CreateChatCompletionRequest:
        unsupported = {"stream", "stream_options", "tools", "tool_choice"} & set(argument.kwargs)
        if unsupported:
            msg = (
                "DeepSeek integration does not support these request options: "
                f"{sorted(unsupported)}"
            )
            raise ValueError(msg)

        request_kwargs = set(CreateChatCompletionRequest.model_fields) - {"messages", "model"}
        payload = self.collect_request_kwargs(argument, request_kwargs)
        if isinstance(payload.get("thinking"), dict):
            payload["thinking"] = Thinking.model_validate(payload["thinking"], strict=False)
        if isinstance(payload.get("reasoning_effort"), str):
            payload["reasoning_effort"] = ReasoningEffort(payload["reasoning_effort"])
        payload["model"] = self.model
        payload["messages"] = tuple(argument.prop.prepared_input)
        return CreateChatCompletionRequest.model_validate(payload)

    def forward(self, argument):  # pyright: ignore[reportIncompatibleMethodOverride]
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def call_request(self, request: CreateChatCompletionRequest) -> APIResponse[ChatCompletion]:
        return self.client.create_chat_completion(request)

    def parse_response(self, response: APIResponse[ChatCompletion]):
        raw_output = response.data
        choice = raw_output.choices[0]
        reasoning_content = choice.message.reasoning_content
        content = choice.message.content or ""
        metadata = {
            "raw_output": raw_output,
            "response": response,
            "thinking": reasoning_content,
        }

        return [content], metadata

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord:
        usage = metadata["raw_output"].usage
        completion_details = usage.completion_tokens_details
        reasoning_tokens = (
            completion_details.reasoning_tokens if completion_details is not None else 0
        )

        return EngineUsageRecord(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            completion_breakdown={"reasoning_tokens": reasoning_tokens or 0},
        )

    def prepare(self, argument):
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        system = render_chat_system_prompt(argument)
        user_prompt = self._build_user_prompt(argument)
        system, user_prompt = self._apply_self_prompt(argument, system, user_prompt)

        argument.prop.prepared_input = [
            {"role": "system", "content": system},
            user_prompt,
        ]

    def _apply_self_prompt(self, argument, system, user_prompt):
        prop = argument.prop
        if prop.instance._kwargs.get("self_prompt", False) or prop.self_prompt:
            res = self.self_prompt({"user": user_prompt["content"], "system": system})
            if res is None:
                msg = "Self-prompting failed for the DeepSeek language-model engine."
                raise ValueError(msg)

            user_prompt = {"role": "user", "content": res["user"]}
            system = res["system"]

        return system, user_prompt

    def _prepare_raw_input(self, argument):
        value = argument.prop.processed_input
        if not value:
            msg = (
                "A prompt instruction is required for the DeepSeek language-model engine "
                "when raw_input is enabled."
            )
            raise ValueError(msg)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [{"role": "user", "content": str(value)}]

    def _build_user_prompt(self, argument):
        return {"role": "user", "content": f"{argument.prop.processed_input!s}"}
