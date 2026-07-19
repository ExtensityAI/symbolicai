from __future__ import annotations

import json
import logging
import re
from copy import deepcopy

from symai.backend.base import Engine
from symai.backend.engines.neurosymbolic._prompts import render_chat_system_prompt
from symai.backend.engines.neurosymbolic.groq.models import (
    GROQ_UNSUPPORTED_REQUEST_KWARGS,
    SUPPORTED_GROQ_MODELS,
    GroqMessage,
    GroqOptions,
    GroqPayload,
    GroqRequest,
    GroqResponse,
    groq_model_spec_for,
    groq_strip_prefix,
)
from symai.backend.engines.neurosymbolic.groq.stream import GroqStreamAdapter
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

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqEngine(Engine):
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
        self.max_response_tokens = self.api_max_response_tokens()
        self.transport_client = None

    def id(self) -> str:
        if self.model in SUPPORTED_GROQ_MODELS and self.api_key:
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def api_max_context_tokens(self) -> int:
        return groq_model_spec_for(self.model).context_tokens

    def api_max_response_tokens(self) -> int:
        return groq_model_spec_for(self.model).response_tokens

    def compute_required_tokens(self, _messages: list[GroqMessage]) -> int:
        msg = 'Method "compute_required_tokens" not implemented for GroqEngine.'
        raise NotImplementedError(msg)

    def compute_remaining_tokens(self, _prompts: list[GroqMessage]) -> int:
        msg = 'Method "compute_remaining_tokens" not implemented for GroqEngine.'
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
        )

    def build_request(self, argument) -> GroqRequest:
        allowed_request_kwargs = (
            set(GroqPayload.model_fields)
            | set(GroqOptions.model_fields)
            | {"max_tokens"}
            | set(GROQ_UNSUPPORTED_REQUEST_KWARGS)
        )
        payload_kwargs = self.collect_request_kwargs(argument, allowed_request_kwargs)
        option_kwargs = {
            key: payload_kwargs.pop(key)
            for key in GroqOptions.model_fields
            if key in payload_kwargs
        }
        for key in sorted(GROQ_UNSUPPORTED_REQUEST_KWARGS):
            if key in payload_kwargs:
                logger.warning(
                    "The parameter %s is not supported by the Groq API. It will be ignored.",
                    key,
                )
                payload_kwargs.pop(key)

        model = groq_strip_prefix(payload_kwargs.get("model", self.model))
        spec = groq_model_spec_for(model)

        payload_kwargs["model"] = model
        payload_kwargs["messages"] = strip_cache_breakpoints_from_messages(
            argument.prop.prepared_input
        )
        # NOTE: Groq deprecated max_tokens in favor of max_completion_tokens; accept the
        # legacy kwarg and remap it when the new one is absent.
        max_tokens = payload_kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_completion_tokens" not in payload_kwargs:
            payload_kwargs["max_completion_tokens"] = max_tokens
        # NOTE: core decorators default stop="" (meaning unset); an empty stop on the wire
        # truncates generation at the first character. Groq omits stop when unset.
        if not payload_kwargs.get("stop"):
            payload_kwargs["stop"] = None

        n = payload_kwargs.get("n", 1)
        if n > 1:
            logger.warning(
                "If N is supplied, it must be equal to 1. We default to 1 to not crash your program."
            )
            n = 1
        payload_kwargs["n"] = n

        if payload_kwargs.get("stream"):
            # NOTE: usage is required on GroqResponse (MetadataTracker reads it), and
            # streams only carry usage in the final chunk when include_usage is set.
            payload_kwargs.setdefault("stream_options", {"include_usage": True})

        if "reasoning_effort" not in payload_kwargs:
            payload_kwargs["reasoning_effort"] = spec.default_reasoning_effort
        if payload_kwargs["reasoning_effort"] not in spec.reasoning_efforts:
            msg = (
                f"Unsupported reasoning_effort for Groq model {model}: "
                f"{payload_kwargs['reasoning_effort']}"
            )
            raise ValueError(msg)

        # NOTE: Groq's json_object mode conflicts with tool use; the API expects tools to
        # be absent and tool_choice to allow generation when JSON mode is requested.
        response_format = payload_kwargs.get("response_format")
        tools = payload_kwargs.get("tools")
        tool_choice = payload_kwargs.get("tool_choice", "auto" if tools else "none")
        if (
            response_format
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        ):
            if tool_choice in (None, "none"):
                tool_choice = "auto"
            if tools:
                tools = None
        payload_kwargs["tools"] = tools
        payload_kwargs["tool_choice"] = tool_choice

        payload = GroqPayload.model_validate(payload_kwargs)
        options = GroqOptions.model_validate(option_kwargs)
        request_options = options.model_dump(exclude_none=True)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(request_options.get("extra_headers", {}))

        return GroqRequest(
            provider="groq",
            operation="chat.completions.create",
            payload=payload,
            call_options=options,
            method="POST",
            url=GROQ_CHAT_COMPLETIONS_URL,
            headers=headers,
            params=request_options.get("extra_query"),
            timeout=request_options.get("timeout", self.client_timeout),
            extra_body=request_options.get("extra_body"),
        )

    def forward(self, argument):
        if self.id() != "neurosymbolic":
            msg = (
                "Groq engine is not configured. Please set a supported "
                "NEUROSYMBOLIC_ENGINE_MODEL and NEUROSYMBOLIC_ENGINE_API_KEY."
            )
            raise ValueError(msg)

        request = self.build_request(argument)
        except_remedy = argument.kwargs.get("except_remedy")
        try:
            response = self.call_request(request)
        except Exception as e:
            if except_remedy is None:
                raise
            # NOTE: the legacy engine passed the SDK callable as `callback`; the
            # raw-REST engine retries the wire request through this closure instead.
            response = except_remedy(
                self, e, lambda *_args, **_kwargs: self.call_request(request), argument
            )
        return self.parse_response(response)

    def call_request(self, request: GroqRequest):
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
        return GroqResponse.model_validate(response.json())

    def parse_response(self, response: GroqResponse):
        metadata: dict = {"raw_output": response}
        metadata = self._process_function_calls(response, metadata)

        outputs = [choice.message.content or "" for choice in response.choices]
        thinking_content = None
        for choice in response.choices:
            if thinking_content is None and choice.message.reasoning:
                thinking_content = choice.message.reasoning

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
                msg = "Self-prompting failed for GroqEngine."
                raise ValueError(msg)

            user_prompt = {"role": "user", "content": res["user"]}
            system = res["system"]

        return system, user_prompt

    def _prepare_raw_input(self, argument):
        value = argument.prop.processed_input
        if not value:
            msg = "A prompt instruction is required for GroqEngine when raw_input is enabled."
            raise ValueError(msg)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [{"role": "user", "content": str(value)}]

    def _collect_stream_response(self, request: GroqRequest, max_retries: int):
        adapter = GroqStreamAdapter()
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

        return GroqResponse.model_validate(
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

    def _extract_thinking_content(self, outputs, thinking_content=None):
        # NOTE: matches one <think>...</think> block (including newlines) so reasoning
        # emitted inline can be separated from the user-facing answer.
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

    def _process_function_calls(self, response: GroqResponse, metadata: dict) -> dict:
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
