from __future__ import annotations

from copy import deepcopy
from typing import Any

import httpx

from symai.backend.base import Engine
from symai.backend.chat_prompts import render_chat_system_prompt
from symai.backend.integrations.deepseek.chat import (
    ChatCompletion,
    CreateChatCompletionRequest,
    Thinking,
)
from symai.backend.integrations.deepseek.client import Client as DeepSeekClient
from symai.backend.integrations.deepseek.transport import APIResponse  # noqa: TC001
from symai.backend.mixin.deepseek import SUPPORTED_MODELS, DeepSeekMixin
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.usage import EngineUsageRecord

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekEngine(Engine, DeepSeekMixin):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
        http_client: httpx.Client | None = None,
    ):
        super().__init__(client_timeout=client_timeout, client_max_retries=client_max_retries)
        self.name = self.__class__.__name__
        self.config = deepcopy(SYMAI_CONFIG)
        if api_key is not None:
            self.config["NEUROSYMBOLIC_ENGINE_API_KEY"] = api_key
        if model is not None:
            self.config["NEUROSYMBOLIC_ENGINE_MODEL"] = model
        self.api_key = self.config["NEUROSYMBOLIC_ENGINE_API_KEY"]
        self.model = self.config["NEUROSYMBOLIC_ENGINE_MODEL"]
        if self.id() != "neurosymbolic":
            return
        self.tokenizer = None
        self.max_context_tokens = self.api_max_context_tokens()
        self.http_client = http_client

    def id(self) -> str:
        if self.model in SUPPORTED_MODELS and self.api_key:
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def compute_required_tokens(self, _messages: list[dict[str, Any]]) -> int:
        msg = 'Method "compute_required_tokens" not implemented for DeepSeekEngine.'
        raise NotImplementedError(msg)

    def compute_remaining_tokens(self, _prompts: list[dict[str, Any]]) -> int:
        msg = 'Method "compute_remaining_tokens" not implemented for DeepSeekEngine.'
        raise NotImplementedError(msg)

    def build_request(self, argument) -> CreateChatCompletionRequest:
        unsupported = {"stream", "stream_options", "tools", "tool_choice"} & set(argument.kwargs)
        if unsupported:
            msg = (
                "DeepSeek integration does not support these request options: "
                f"{sorted(unsupported)}"
            )
            raise ValueError(msg)

        request_kwargs = set(CreateChatCompletionRequest.model_fields) - {"messages"}
        payload = self.collect_request_kwargs(argument, request_kwargs)
        if isinstance(payload.get("thinking"), dict):
            payload["thinking"] = Thinking.model_validate(payload["thinking"], strict=False)
        payload["model"] = payload.get("model", self.model)
        self.deepseek_model_spec_for(payload["model"])
        payload["messages"] = tuple(argument.prop.prepared_input)
        return CreateChatCompletionRequest.model_validate(payload)

    def forward(self, argument):  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.id() != "neurosymbolic":
            msg = (
                "DeepSeek engine is not configured. Please set a supported "
                "NEUROSYMBOLIC_ENGINE_MODEL and NEUROSYMBOLIC_ENGINE_API_KEY."
            )
            raise ValueError(msg)

        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def call_request(self, request: CreateChatCompletionRequest) -> APIResponse[ChatCompletion]:
        if self.http_client is not None:
            return DeepSeekClient(
                api_key=self.api_key,
                http_client=self.http_client,
            ).create_chat_completion(request)

        with httpx.Client(timeout=self.client_timeout) as http_client:
            return DeepSeekClient(
                api_key=self.api_key,
                http_client=http_client,
            ).create_chat_completion(request)

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
                msg = "Self-prompting failed for DeepSeekEngine."
                raise ValueError(msg)

            user_prompt = {"role": "user", "content": res["user"]}
            system = res["system"]

        return system, user_prompt

    def _prepare_raw_input(self, argument):
        value = argument.prop.processed_input
        if not value:
            msg = "A prompt instruction is required for DeepSeekEngine when raw_input is enabled."
            raise ValueError(msg)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [{"role": "user", "content": str(value)}]

    def _build_user_prompt(self, argument):
        return {"role": "user", "content": f"{argument.prop.processed_input!s}"}
