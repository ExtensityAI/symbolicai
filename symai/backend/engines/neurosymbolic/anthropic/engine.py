from __future__ import annotations

import json
import logging
import re
from copy import deepcopy

from symai.backend.base import Engine
from symai.backend.engines.neurosymbolic._prompts import render_chat_system_prompt
from symai.backend.engines.neurosymbolic.anthropic.models import (
    ANTHROPIC_VERSION,
    LONG_CONTEXT_1M_BETA_HEADER,
    SUPPORTED_ANTHROPIC_MODELS,
    AnthropicCountTokensPayload,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicOptions,
    AnthropicPayload,
    AnthropicRequest,
    AnthropicResponse,
    anthropic_model_spec_for,
    anthropic_strip_prefix,
    build_cache_breakpoint_blocks,
    resolve_cache_control,
)
from symai.backend.engines.neurosymbolic.anthropic.stream import AnthropicStreamAdapter
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.streaming import EngineStreamAccumulator
from symai.backend.transport import (
    DEFAULT_RETRIES,
    execute_engine_api_request,
    execute_engine_api_stream_events,
)
from symai.backend.usage import EngineUsageRecord
from symai.prompts import CACHE_BREAKPOINT, strip_cache_breakpoints
from symai.utils import encode_media_frames

logger = logging.getLogger(__name__)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"


class TokenizerWrapper:
    def __init__(self, compute_tokens_func):
        self.compute_tokens_func = compute_tokens_func

    def encode(self, text: str) -> int:
        return self.compute_tokens_func([{"role": "user", "content": text}])


class AnthropicEngine(Engine):
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
        self.tokenizer = TokenizerWrapper(self.compute_required_tokens)
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.transport_client = None

    def id(self) -> str:
        if self.model in SUPPORTED_ANTHROPIC_MODELS and self.api_key:
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def api_max_context_tokens(self) -> int:
        return anthropic_model_spec_for(self.model).context_tokens

    def api_max_response_tokens(self) -> int:
        return anthropic_model_spec_for(self.model).response_tokens

    def is_reasoning_model(self) -> bool:
        return anthropic_model_spec_for(self.model).reasoning

    def compute_required_tokens(self, messages: list[dict]) -> int:
        system, wire_messages = self._build_wire_messages(messages)
        if not wire_messages:
            return 0

        payload = {"model": anthropic_strip_prefix(self.model), "messages": wire_messages}
        if system:
            payload["system"] = system
        request = AnthropicCountTokensRequest(
            provider="anthropic",
            operation="messages.count_tokens",
            payload=AnthropicCountTokensPayload.model_validate(payload),
            method="POST",
            url=ANTHROPIC_COUNT_TOKENS_URL,
            headers=self._auth_headers(),
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=self.client_max_retries
            if self.client_max_retries is not None
            else DEFAULT_RETRIES,
        )
        return AnthropicCountTokensResponse.model_validate(response.json()).input_tokens

    def compute_remaining_tokens(self, _prompts: list[dict]) -> int:
        msg = 'Method "compute_remaining_tokens" not implemented for AnthropicEngine.'
        raise NotImplementedError(msg)

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord:
        usage = metadata["raw_output"].usage

        return EngineUsageRecord(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            prompt_breakdown={
                "cached_tokens": usage.cache_read_input_tokens or 0,
            },
            extras={
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            },
        )

    def build_request(self, argument) -> AnthropicRequest:
        allowed_request_kwargs = (
            set(AnthropicPayload.model_fields)
            | set(AnthropicOptions.model_fields)
            # symai kwargs handled outside the payload models: "stop" aliases the
            # stop_sequences wire field; "response_format" feeds _build_output_config.
            | {"stop", "response_format"}
        )
        payload_kwargs = self.collect_request_kwargs(argument, allowed_request_kwargs)
        option_kwargs = {
            key: payload_kwargs.pop(key)
            for key in AnthropicOptions.model_fields
            if key in payload_kwargs
        }

        model = anthropic_strip_prefix(payload_kwargs.get("model", self.model))
        spec = anthropic_model_spec_for(model)
        system, messages = self._build_wire_messages(argument.prop.prepared_input)

        cache_control = resolve_cache_control(payload_kwargs.pop("cache_control", None))
        messages, cache_control = self._apply_cache_breakpoints(messages, cache_control)

        long_context_1m = payload_kwargs.pop("long_context_1m", False)
        extra_beta_headers = {}
        if long_context_1m:
            if spec.long_context_1m and not spec.default_long_context_1m:
                extra_beta_headers = {"anthropic-beta": LONG_CONTEXT_1M_BETA_HEADER}
            elif not spec.long_context_1m:
                logger.warning(
                    "long_context_1m is not supported by %s; falling back to %s token context.",
                    model,
                    spec.context_tokens,
                )

        thinking, adaptive_effort = self._build_thinking_config(
            payload_kwargs.pop("thinking", None), model, spec
        )
        output_config = self._build_output_config(payload_kwargs.pop("response_format", None))
        output_config = self._merge_output_config_effort(output_config, adaptive_effort)

        stop = payload_kwargs.pop("stop", None)
        if not stop:
            stop = None
        elif isinstance(stop, str):
            stop = [stop]

        if not spec.sampling:
            payload_kwargs.pop("temperature", None)
            payload_kwargs.pop("top_p", None)
            payload_kwargs.pop("top_k", None)

        payload_kwargs["model"] = model
        payload_kwargs["messages"] = messages
        if system:
            payload_kwargs["system"] = system
        payload_kwargs.setdefault("max_tokens", spec.response_tokens)
        if thinking is not None:
            payload_kwargs["thinking"] = thinking
        if output_config is not None:
            payload_kwargs["output_config"] = output_config
        payload_kwargs["stop_sequences"] = stop
        # Do NOT remove this default value! Getting tons of API errors because they
        # can't process requests >10m.
        payload_kwargs.setdefault("stream", True)

        payload = AnthropicPayload.model_validate(payload_kwargs)
        options = AnthropicOptions.model_validate(option_kwargs)
        request_options = options.model_dump(exclude_none=True)

        headers = self._auth_headers()
        headers.update(extra_beta_headers)
        headers.update(request_options.get("extra_headers", {}))

        extra_body = request_options.get("extra_body")
        if cache_control is not None:
            extra_body = {**(extra_body or {}), "cache_control": cache_control}

        return AnthropicRequest(
            provider="anthropic",
            operation="messages.create",
            payload=payload,
            call_options=options,
            method="POST",
            url=ANTHROPIC_MESSAGES_URL,
            headers=headers,
            params=request_options.get("extra_query"),
            timeout=request_options.get("timeout", self.client_timeout),
            extra_body=extra_body,
        )

    def forward(self, argument):
        if self.id() != "neurosymbolic":
            msg = (
                "Anthropic engine is not configured. Please set a supported "
                "NEUROSYMBOLIC_ENGINE_MODEL and NEUROSYMBOLIC_ENGINE_API_KEY."
            )
            raise ValueError(msg)

        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response, argument)

    def call_request(self, request: AnthropicRequest):
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
        return AnthropicResponse.model_validate(response.json())

    def parse_response(self, response: AnthropicResponse, argument=None):
        metadata: dict = {"raw_output": response}
        metadata = self._process_function_calls(response, metadata)

        outputs = []
        thinking_parts = []
        for block in response.content:
            if block.type == "text" and block.text is not None:
                outputs.append(block.text)
            elif block.type == "thinking" and block.thinking:
                thinking_parts.append(block.thinking)

        metadata["thinking"] = "\n".join(thinking_parts) or None

        text_output = "".join(outputs)
        if argument is not None and getattr(argument.prop, "response_format", None):
            # NOTE: Anthropic returns JSON wrapped in markdown fences.
            text_output = text_output.replace("```json", "").replace("```", "")

        if not text_output and "function_call" in metadata:
            text_output = ""

        return [text_output], metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        image_files = self._handle_image_content(str(argument.prop.processed_input))
        system = render_chat_system_prompt(argument)
        user_text = self._build_user_text(argument, image_files)
        if not user_text:
            # NOTE: Anthropic rejects empty user prompts.
            user_text = "N/A"

        user_prompt = self._format_user_prompt(user_text, image_files)
        system, user_prompt = self._apply_self_prompt(
            argument, system, user_prompt, user_text, image_files
        )

        argument.prop.prepared_input = [
            {"role": "system", "content": system},
            user_prompt,
        ]

    def _apply_self_prompt(self, argument, system, user_prompt, user_text, image_files):
        prop = argument.prop
        if not (prop.instance._kwargs.get("self_prompt", False) or prop.self_prompt):
            return system, user_prompt

        res = self.self_prompt(
            {"user": user_text, "system": system},
            max_tokens=argument.kwargs.get("max_tokens", self.max_response_tokens),
            **({"thinking": argument.kwargs["thinking"]} if "thinking" in argument.kwargs else {}),
        )
        if res is None:
            msg = "Self-prompting failed for AnthropicEngine."
            raise ValueError(msg)

        return res["system"], self._format_user_prompt(res["user"], image_files)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _build_wire_messages(self, prepared_input) -> tuple[str | None, list[dict]]:
        """Split uniform prepared input into Anthropic's top-level system + messages."""
        system = None
        messages = []
        for message in prepared_input:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                system = content if isinstance(content, str) else json.dumps(content)
                continue
            messages.append(self._build_message_payload(role, content))
        return system, messages

    def _build_message_payload(self, role, content) -> dict:
        if not isinstance(content, str):
            return {"role": role, "content": content}

        image_blocks = self._handle_image_content(content)
        text = self._remove_vision_pattern(content)
        if not image_blocks:
            return {"role": role, "content": text}

        blocks = [{"type": "image", "source": image} for image in image_blocks]
        if text:
            blocks.append({"type": "text", "text": text})
        return {"role": role, "content": blocks}

    def _apply_cache_breakpoints(self, messages: list[dict], cache_control):
        """Translate ``symai:cache_breakpoint`` markers into cache_control blocks.

        Block-level breakpoints supersede the top-level auto-cache form, so a split
        clears it. With caching disabled or no markers, the marker is stripped.
        """
        did_split = False
        new_messages = []
        for message in messages:
            images, text = self._extract_message_text(message.get("content"))
            if text is None or CACHE_BREAKPOINT not in text:
                new_messages.append(message)
                continue
            if cache_control is not None:
                blocks = build_cache_breakpoint_blocks(text, cache_control)
                new_messages.append({**message, "content": [*images, *blocks]})
                did_split = True
            else:
                stripped = strip_cache_breakpoints(text)
                content = stripped if not images else [*images, {"type": "text", "text": stripped}]
                new_messages.append({**message, "content": content})
        if did_split:
            cache_control = None
        return new_messages, cache_control

    @staticmethod
    def _extract_message_text(content):
        if isinstance(content, str):
            return [], content
        if isinstance(content, list):
            images = [
                block
                for block in content
                if not (isinstance(block, dict) and block.get("type") == "text")
            ]
            texts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return images, "".join(texts)
        return [], None

    def _build_thinking_config(self, thinking_arg, model, spec):
        if not thinking_arg or not isinstance(thinking_arg, dict):
            return None, None

        thinking_type = thinking_arg.get("type")
        if thinking_type == "disabled":
            return {"type": "disabled"}, None

        if model in {"claude-opus-4-8", "claude-opus-4-7"}:
            return {"type": "adaptive"}, thinking_arg.get("effort")

        if thinking_type == "adaptive":
            if spec.adaptive_thinking:
                return {"type": "adaptive"}, thinking_arg.get("effort")
            logger.warning(
                "Adaptive thinking is only supported for claude-opus-4-8, claude-opus-4-7, claude-opus-4-6 and claude-sonnet-4-6; "
                "falling back to manual thinking."
            )
            return {
                "type": "enabled",
                "budget_tokens": thinking_arg.get("budget_tokens", 1024),
            }, None

        if thinking_type == "enabled" or "budget_tokens" in thinking_arg:
            return {
                "type": "enabled",
                "budget_tokens": thinking_arg.get("budget_tokens", 1024),
            }, None

        return None, None

    def _build_output_config(self, response_format):
        if not isinstance(response_format, dict):
            return None

        if response_format.get("type") == "json_schema":
            schema = response_format.get("schema")
            if schema is None and response_format.get("json_schema") is not None:
                schema = response_format["json_schema"].get(
                    "schema", response_format["json_schema"]
                )
            if schema is None:
                return None
            return {"format": {"type": "json_schema", "schema": schema}}

        return None

    def _merge_output_config_effort(self, output_config, adaptive_effort):
        if adaptive_effort is None:
            return output_config
        if output_config is None:
            return {"effort": adaptive_effort}
        return {**output_config, "effort": adaptive_effort}

    def _collect_stream_response(self, request: AnthropicRequest, max_retries: int):
        adapter = AnthropicStreamAdapter()
        accumulator = EngineStreamAccumulator()
        usage = {}

        for event in execute_engine_api_stream_events(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        ):
            delta = adapter.process_event(event)
            accumulator.add(delta)
            if delta.usage:
                # NOTE: Anthropic splits usage across message_start (input) and
                # message_delta (output); merge instead of keeping only the last.
                usage.update(delta.usage)
            if accumulator.done:
                break

        content = []
        if accumulator.thinking:
            content.append({"type": "thinking", "thinking": accumulator.thinking})
        content.append({"type": "text", "text": accumulator.text})
        for tool_call in adapter.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": tool_call["input"],
                }
            )

        return AnthropicResponse.model_validate(
            {
                "role": "assistant",
                "content": content,
                "stop_reason": accumulator.finish_reason,
                "usage": usage,
            }
        )

    def _handle_image_content(self, content: str) -> list:
        # NOTE: matches <<vision:...:>> markers embedding image references in the prompt.
        if "<<vision:" not in content:
            return []

        image_files = []
        for p in re.findall(r"<<vision:(.*?):>>", content):
            img_ = p.strip()
            max_frames_spacing = 50
            max_used_frames = 10
            buffer, ext = encode_media_frames(img_)
            if len(buffer) > 1:
                step = max(1, len(buffer) // max_frames_spacing)
                indices = list(range(0, len(buffer), step))[:max_used_frames]
                for i in indices:
                    image_files.append(
                        {"data": buffer[i], "media_type": f"image/{ext}", "type": "base64"}
                    )
            elif len(buffer) == 1:
                image_files.append(
                    {"data": buffer[0], "media_type": f"image/{ext}", "type": "base64"}
                )
            else:
                logger.warning("No frames found for image!")
        return image_files

    def _remove_vision_pattern(self, text: str) -> str:
        # NOTE: strips <<vision:...:>> markers from text after image extraction.
        return re.sub(r"<<vision:(.*?):>>", "", text)

    def _build_user_text(self, argument, image_files: list) -> str:
        suffix = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = self._remove_vision_pattern(suffix)
        return suffix

    def _format_user_prompt(self, user_text: str, image_files: list) -> dict:
        if len(image_files) > 0:
            images = [{"type": "image", "source": im} for im in image_files]
            return {"role": "user", "content": [*images, {"type": "text", "text": user_text}]}
        return {"role": "user", "content": user_text}

    def _prepare_raw_input(self, argument):
        value = argument.prop.processed_input
        if not value:
            msg = "A prompt instruction is required for AnthropicEngine when raw_input is enabled."
            raise ValueError(msg)
        if not isinstance(value, list):
            value = [value]
        system = None
        prompt = []
        for part in value:
            item = part
            if isinstance(item, dict) and item.get("role") == "system":
                system = item["content"]
                continue
            if isinstance(item, str):
                item = {"role": "user", "content": item}
            prompt.append(item)
        if system is not None:
            return [{"role": "system", "content": system}, *prompt]
        return prompt

    def _process_function_calls(self, response: AnthropicResponse, metadata: dict) -> dict:
        for block in response.content:
            if block.type == "tool_use":
                if "function_call" in metadata:
                    logger.warning(
                        "Multiple tool use blocks detected in the response but only the first one will be processed."
                    )
                    break
                metadata["function_call"] = {
                    "name": block.name,
                    "arguments": block.input or {},
                    "id": block.id,
                }
        return metadata
