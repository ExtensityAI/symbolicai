from __future__ import annotations

import json
import logging
from copy import deepcopy

from symai.backend.base import Engine
from symai.backend.engines.neurosymbolic.deepseek.models import (
    SUPPORTED_MODELS,
    DeepSeekMessage,
    DeepSeekOptions,
    DeepSeekPayload,
    DeepSeekRequest,
    DeepSeekResponse,
    deepseek_model_spec_for,
    deepseek_strip_prefix,
)
from symai.backend.engines.neurosymbolic.deepseek.stream import DeepSeekStreamAdapter
from symai.backend.engines.neurosymbolic.prompts import render_chat_system_prompt
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.streaming import EngineStreamAccumulator
from symai.backend.transport import (
    DEFAULT_RETRIES,
    execute_engine_api_request,
    execute_engine_api_stream_events,
)
from symai.backend.usage import EngineUsageRecord
from symai.prompts import strip_cache_breakpoints_from_messages

logger = logging.getLogger(__name__)

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"


class DeepseekEngine(Engine):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
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
        self.transport_client = None

    def id(self) -> str:
        if self.model in SUPPORTED_MODELS and self.api_key:
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def api_max_context_tokens(self) -> int:
        return deepseek_model_spec_for(self.model).context_tokens

    def compute_required_tokens(self, _messages: list[DeepSeekMessage]) -> int:
        msg = 'Method "compute_required_tokens" not implemented for DeepseekEngine.'
        raise NotImplementedError(msg)

    def compute_remaining_tokens(self, _prompts: list[DeepSeekMessage]) -> int:
        msg = 'Method "compute_remaining_tokens" not implemented for DeepseekEngine.'
        raise NotImplementedError(msg)

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord:
        usage = metadata["raw_output"].usage
        completion_details = usage.completion_tokens_details

        return EngineUsageRecord(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            completion_breakdown={
                "reasoning_tokens": (completion_details.reasoning_tokens or 0)
                if completion_details
                else 0,
            },
            extras={
                "prompt_cache_hit_tokens": usage.prompt_cache_hit_tokens,
                "prompt_cache_miss_tokens": usage.prompt_cache_miss_tokens,
            },
        )

    def build_request(self, argument) -> DeepSeekRequest:
        allowed_request_kwargs = set(DeepSeekPayload.model_fields).union(
            DeepSeekOptions.model_fields
        )
        payload_kwargs = self.collect_request_kwargs(argument, allowed_request_kwargs)
        option_kwargs = {
            key: payload_kwargs.pop(key)
            for key in DeepSeekOptions.model_fields
            if key in payload_kwargs
        }
        payload_kwargs["model"] = deepseek_strip_prefix(payload_kwargs.get("model", self.model))
        deepseek_model_spec_for(payload_kwargs["model"])
        payload_kwargs["messages"] = strip_cache_breakpoints_from_messages(
            argument.prop.prepared_input
        )
        # NOTE: every core decorator signature defaults stop="" (meaning unset), so an
        # empty stop always arrives via argument.kwargs. Passing "" to the API truncates
        # generation at the first character, so only a non-empty user stop wins over the default.
        if not payload_kwargs.get("stop"):
            payload_kwargs["stop"] = "<|endoftext|>"
        if payload_kwargs.get("stream"):
            # NOTE: usage is required on DeepSeekResponse (MetadataTracker reads it), and
            # streams only carry usage in the final chunk when include_usage is set.
            payload_kwargs.setdefault("stream_options", {"include_usage": True})

        payload = DeepSeekPayload.model_validate(payload_kwargs)
        options = DeepSeekOptions.model_validate(option_kwargs)
        request_options = options.model_dump(exclude_none=True)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(request_options.get("extra_headers", {}))

        return DeepSeekRequest(
            provider="deepseek",
            operation="chat.completions.create",
            payload=payload,
            call_options=options,
            method="POST",
            url=DEEPSEEK_CHAT_COMPLETIONS_URL,
            headers=headers,
            params=request_options.get("extra_query"),
            timeout=request_options.get("timeout", self.client_timeout),
            extra_body=request_options.get("extra_body"),
        )

    def forward(self, argument):
        if self.id() != "neurosymbolic":
            msg = (
                "DeepSeek engine is not configured. Please set a supported "
                "NEUROSYMBOLIC_ENGINE_MODEL and NEUROSYMBOLIC_ENGINE_API_KEY."
            )
            raise ValueError(msg)

        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def call_request(self, request: DeepSeekRequest):
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        if request.payload.stream:
            return self._collect_stream_response(request, max_retries)
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        return DeepSeekResponse.model_validate(response.json())

    def parse_response(self, response: DeepSeekResponse):
        metadata = {
            "raw_output": response,
            "thinking": response.choices[0].message.reasoning_content,
        }
        metadata = self._process_function_calls(response, metadata)

        return [response.choices[0].message.content or ""], metadata

    def _process_function_calls(self, response: DeepSeekResponse, metadata: dict) -> dict:
        message = response.choices[0].message
        if not message.tool_calls:
            return metadata

        for tool_call in message.tool_calls:
            if tool_call.function is None:
                continue
            if "function_call" in metadata:
                logger.warning(
                    "Multiple function calls detected in the response but only the first one will be processed."
                )
                break
            try:
                args_dict = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args_dict = {}
            metadata["function_call"] = {
                "name": tool_call.function.name,
                "arguments": args_dict,
            }

        return metadata

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
                msg = "Self-prompting failed for DeepseekEngine."
                raise ValueError(msg)

            user_prompt = {"role": "user", "content": res["user"]}
            system = res["system"]

        return system, user_prompt

    def _collect_stream_response(self, request: DeepSeekRequest, max_retries: int):
        adapter = DeepSeekStreamAdapter()
        accumulator = EngineStreamAccumulator()

        for event in execute_engine_api_stream_events(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        ):
            delta = adapter.process_event(event)
            accumulator.add(delta)
            if accumulator.done:
                break

        message = {"role": "assistant", "content": accumulator.text}
        if accumulator.thinking:
            message["reasoning_content"] = accumulator.thinking

        return DeepSeekResponse.model_validate(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": accumulator.finish_reason,
                    }
                ],
                "usage": accumulator.usage,
            }
        )

    def _prepare_raw_input(self, argument):
        value = argument.prop.processed_input
        if not value:
            msg = "A prompt instruction is required for DeepseekEngine when raw_input is enabled."
            raise ValueError(msg)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [{"role": "user", "content": str(value)}]

    def _build_user_prompt(self, argument):
        return {"role": "user", "content": f"{argument.prop.processed_input!s}"}
