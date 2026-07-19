from __future__ import annotations

import json
import logging
import re
from copy import deepcopy

from symai.backend.base import Engine
from symai.backend.engines.neurosymbolic._prompts import render_chat_system_prompt
from symai.backend.engines.neurosymbolic.cerebras.models import (
    SUPPORTED_CEREBRAS_MODELS,
    CerebrasOptions,
    CerebrasPayload,
    CerebrasRequest,
    CerebrasResponse,
    cerebras_model_spec_for,
    cerebras_normalize_model,
    cerebras_strip_prefix,
)
from symai.backend.engines.neurosymbolic.cerebras.stream import CerebrasStreamAdapter
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

CEREBRAS_CHAT_COMPLETIONS_URL = "https://api.cerebras.ai/v1/chat/completions"


class CerebrasEngine(Engine):
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
        self.model = cerebras_normalize_model(self.config["NEUROSYMBOLIC_ENGINE_MODEL"])
        if self.id() != "neurosymbolic":
            return
        self.tokenizer = None
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.transport_client = None

    def id(self) -> str:
        if self.model in SUPPORTED_CEREBRAS_MODELS and self.api_key:
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def api_max_context_tokens(self) -> int:
        return cerebras_model_spec_for(self.model).context_tokens

    def api_max_response_tokens(self) -> int:
        return cerebras_model_spec_for(self.model).response_tokens

    def compute_required_tokens(self, _messages: list[dict]) -> int:
        msg = 'Method "compute_required_tokens" not implemented for CerebrasEngine.'
        raise NotImplementedError(msg)

    def compute_remaining_tokens(self, _prompts: list[dict]) -> int:
        msg = 'Method "compute_remaining_tokens" not implemented for CerebrasEngine.'
        raise NotImplementedError(msg)

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord:
        usage = metadata["raw_output"].usage
        completion_details = usage.completion_tokens_details
        prompt_details = usage.prompt_tokens_details

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
                "prompt_cached_tokens": (prompt_details.cached_tokens or 0)
                if prompt_details
                else 0,
            },
        )

    def build_request(self, argument) -> CerebrasRequest:
        allowed_request_kwargs = set(CerebrasPayload.model_fields).union(
            CerebrasOptions.model_fields
        )
        payload_kwargs = self.collect_request_kwargs(argument, allowed_request_kwargs)
        option_kwargs = {
            key: payload_kwargs.pop(key)
            for key in CerebrasOptions.model_fields
            if key in payload_kwargs
        }

        model = cerebras_strip_prefix(payload_kwargs.get("model", self.model))
        model_spec = cerebras_model_spec_for(model)
        reasoning_effort = payload_kwargs.get("reasoning_effort")
        if reasoning_effort is not None and reasoning_effort not in model_spec.reasoning_efforts:
            msg = (
                f"Unsupported reasoning_effort for Cerebras model {model}: "
                f"{reasoning_effort}. Supported values: {list(model_spec.reasoning_efforts)}"
            )
            raise ValueError(msg)
        payload_kwargs["model"] = model
        payload_kwargs["messages"] = strip_cache_breakpoints_from_messages(
            argument.prop.prepared_input
        )
        # NOTE: core decorators default stop="" (meaning unset); an empty stop on the wire
        # truncates generation at the first character. Cerebras omits stop when unset.
        if not payload_kwargs.get("stop"):
            payload_kwargs["stop"] = None
        payload_kwargs["response_format"] = self._normalize_response_format(
            payload_kwargs.get("response_format")
        )

        n = payload_kwargs.get("n", 1)
        if n > 1:
            logger.warning(
                "If N is supplied, it must be equal to 1. We default to 1 to avoid unexpected batch behavior."
            )
            n = 1
        payload_kwargs["n"] = n

        if payload_kwargs.get("stream"):
            # NOTE: usage is required on CerebrasResponse (MetadataTracker reads it), and
            # streams only carry usage in the final chunk when include_usage is set.
            payload_kwargs.setdefault("stream_options", {"include_usage": True})

        payload = CerebrasPayload.model_validate(payload_kwargs)
        options = CerebrasOptions.model_validate(option_kwargs)
        request_options = options.model_dump(exclude_none=True)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(request_options.get("extra_headers", {}))

        return CerebrasRequest(
            provider="cerebras",
            operation="chat.completions.create",
            payload=payload,
            call_options=options,
            method="POST",
            url=CEREBRAS_CHAT_COMPLETIONS_URL,
            headers=headers,
            params=request_options.get("extra_query"),
            timeout=request_options.get("timeout", self.client_timeout),
            extra_body=request_options.get("extra_body"),
        )

    def forward(self, argument):
        if self.id() != "neurosymbolic":
            msg = (
                "Cerebras engine is not configured. Please set a supported "
                "NEUROSYMBOLIC_ENGINE_MODEL and NEUROSYMBOLIC_ENGINE_API_KEY."
            )
            raise ValueError(msg)

        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def call_request(self, request: CerebrasRequest):
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
        return CerebrasResponse.model_validate(response.json())

    def parse_response(self, response: CerebrasResponse):
        metadata: dict = {"raw_output": response}
        metadata = self._process_function_calls(response, metadata)

        outputs: list[str] = []
        thinking_content: str | None = None

        for choice in response.choices:
            message = choice.message
            outputs.append(message.content or "")
            if thinking_content is None and message.reasoning:
                thinking_content = message.reasoning

        thinking_content, outputs = self._extract_thinking_content(outputs, thinking_content)

        if thinking_content:
            metadata["thinking"] = thinking_content

        if not outputs and "function_call" in metadata:
            outputs = [""]

        return outputs, metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        system = render_chat_system_prompt(argument)
        user_prompt = {"role": "user", "content": str(argument.prop.processed_input)}
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
                msg = "Self-prompting failed for CerebrasEngine."
                raise ValueError(msg)

            user_prompt = {"role": "user", "content": res["user"]}
            system = res["system"]

        return system, user_prompt

    def _collect_stream_response(self, request: CerebrasRequest, max_retries: int):
        adapter = CerebrasStreamAdapter()
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
            message["reasoning"] = accumulator.thinking

        return CerebrasResponse.model_validate(
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
            msg = "A prompt instruction is required for CerebrasEngine when raw_input is enabled."
            raise ValueError(msg)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [{"role": "user", "content": str(value)}]

    @staticmethod
    def _normalize_response_format(response_format: dict | None) -> dict | None:
        # NOTE: Cerebras expects json_schema nested as
        # {"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}
        # but callers may pass the flat form {"type": "json_schema", "name": ..., "schema": ...}.
        if not isinstance(response_format, dict):
            return response_format
        if response_format.get("type") != "json_schema":
            return response_format
        if "json_schema" in response_format:
            return response_format
        inner = {k: v for k, v in response_format.items() if k != "type"}
        inner.setdefault("strict", True)
        return {"type": "json_schema", "json_schema": inner}

    def _extract_thinking_content(self, outputs, thinking_content=None):
        # NOTE: matches one <think>...</think> block (including newlines) so reasoning
        # emitted inline by Cerebras models can be separated from the user-facing answer.
        if not outputs:
            return thinking_content, outputs

        content = outputs[0]
        if not content:
            return thinking_content, outputs

        think_pattern = r"<think>(.*?)</think>"
        match = re.search(think_pattern, content, re.DOTALL)

        if match and thinking_content is None:
            thinking_content = match.group(1).strip() or None

        cleaned_content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()
        cleaned_outputs = [cleaned_content, *outputs[1:]]

        return thinking_content, cleaned_outputs

    def _process_function_calls(self, response: CerebrasResponse, metadata: dict) -> dict:
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
