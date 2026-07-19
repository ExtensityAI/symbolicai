from __future__ import annotations

import json
import logging
import re
from copy import deepcopy

import tiktoken

from symai.backend.base import Engine
from symai.backend.engines.neurosymbolic._prompts import render_chat_system_prompt
from symai.backend.engines.neurosymbolic.openai.models import (
    SUPPORTED_OPENAI_MODELS,
    OpenAIOptions,
    OpenAIPayload,
    OpenAIRequest,
    OpenAIResponse,
    build_cache_breakpoint_blocks,
    openai_model_spec_for,
    openai_strip_prefix,
)
from symai.backend.engines.neurosymbolic.openai.stream import OpenAIStreamAdapter
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.streaming import EngineStreamAccumulator
from symai.backend.transport import (
    DEFAULT_RETRIES,
    execute_engine_api_request,
    execute_engine_api_stream_events,
)
from symai.backend.usage import EngineUsageRecord
from symai.prompts import CACHE_BREAKPOINT
from symai.utils import encode_media_frames

logger = logging.getLogger(__name__)

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class ResponsesTokenizer:
    def __init__(self, model: str, tokenizer_name: str):
        try:
            self._tiktoken = tiktoken.encoding_for_model(model)
        except Exception:
            self._tiktoken = tiktoken.get_encoding(tokenizer_name)

    def encode(self, text: str) -> list[int]:
        return self._tiktoken.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._tiktoken.decode(tokens)


class OpenAIEngine(Engine):
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
        self.tokenizer = ResponsesTokenizer(
            model=openai_strip_prefix(self.model),
            tokenizer_name=openai_model_spec_for(self.model).tokenizer,
        )
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.transport_client = None

    def id(self) -> str:
        if self.model in SUPPORTED_OPENAI_MODELS and self.api_key:
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def api_max_context_tokens(self) -> int:
        return openai_model_spec_for(self.model).context_tokens

    def api_max_response_tokens(self) -> int:
        return openai_model_spec_for(self.model).response_tokens

    def is_reasoning_model(self) -> bool:
        return openai_model_spec_for(self.model).reasoning

    def is_pro_model(self) -> bool:
        return openai_model_spec_for(self.model).pro

    def supports_vision(self) -> bool:
        return openai_model_spec_for(self.model).vision

    def compute_required_tokens(self, messages: list[dict]) -> int:
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(self.tokenizer.encode(value))
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, dict) and v.get("type") in ("text", "input_text"):
                            num_tokens += len(self.tokenizer.encode(v.get("text", "")))
                if key == "name":
                    num_tokens += tokens_per_name
        if self.is_reasoning_model():
            num_tokens += 6
        else:
            num_tokens += 3
        return num_tokens

    def compute_remaining_tokens(self, prompts: list[dict]) -> int:
        val = self.compute_required_tokens(prompts)
        return min(self.max_context_tokens - val, self.max_response_tokens)

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord:
        usage = metadata["raw_output"].usage
        input_details = usage.input_tokens_details
        output_details = usage.output_tokens_details

        return EngineUsageRecord(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            prompt_breakdown={
                "cached_tokens": (input_details.cached_tokens or 0) if input_details else 0,
            },
            completion_breakdown={
                "reasoning_tokens": (output_details.reasoning_tokens or 0) if output_details else 0,
            },
            extras={
                "cache_write_tokens": (input_details.cache_write_tokens or 0)
                if input_details
                else 0,
            },
        )

    def build_request(self, argument) -> OpenAIRequest:
        allowed_request_kwargs = set(OpenAIPayload.model_fields).union(
            OpenAIOptions.model_fields
        ) | {
            # symai kwargs handled outside the payload models: "response_format" maps
            # onto the Responses API text.format field; "max_tokens" is the legacy
            # symai kwarg aliased to max_output_tokens.
            "response_format",
            "max_tokens",
        }
        payload_kwargs = self.collect_request_kwargs(argument, allowed_request_kwargs)
        option_kwargs = {
            key: payload_kwargs.pop(key)
            for key in OpenAIOptions.model_fields
            if key in payload_kwargs
        }

        messages = argument.prop.prepared_input
        payload_kwargs["model"] = openai_strip_prefix(payload_kwargs.get("model", self.model))
        model_spec = openai_model_spec_for(payload_kwargs["model"])
        payload_kwargs["input"] = self._apply_cache_breakpoints(
            messages, model_spec, payload_kwargs["model"]
        )

        # NOTE: the legacy chat-completions engine accepted max_tokens; alias it to the
        # Responses API max_output_tokens when the new kwarg is absent.
        max_tokens = payload_kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_output_tokens" not in payload_kwargs:
            payload_kwargs["max_output_tokens"] = max_tokens

        response_format = self._normalize_response_format(
            payload_kwargs.pop("response_format", None)
        )
        if response_format is not None:
            text = dict(payload_kwargs.get("text") or {})
            text["format"] = response_format
            payload_kwargs["text"] = text

        if self.is_reasoning_model():
            payload_kwargs.pop("temperature", None)
            payload_kwargs.pop("top_p", None)
            if self.is_pro_model():
                payload_kwargs["reasoning"] = {"effort": "high"}
            else:
                payload_kwargs["reasoning"] = payload_kwargs.get("reasoning", {"effort": "medium"})

        tools = payload_kwargs.get("tools")
        if tools:
            payload_kwargs["tools"] = self._convert_tools(tools)
            payload_kwargs["tool_choice"] = payload_kwargs.get("tool_choice", "auto")

        payload = OpenAIPayload.model_validate(payload_kwargs)
        remaining_tokens = self.compute_remaining_tokens(messages)
        max_output_tokens = payload.max_output_tokens

        if max_output_tokens is not None and max_output_tokens > self.max_response_tokens:
            logger.warning(
                "Provided 'max_output_tokens' (%s) exceeds max (%s). Truncating to %s.",
                max_output_tokens,
                self.max_response_tokens,
                remaining_tokens,
            )
            max_output_tokens = remaining_tokens

        if max_output_tokens != payload.max_output_tokens:
            payload_kwargs["max_output_tokens"] = max_output_tokens
            payload = OpenAIPayload.model_validate(payload_kwargs)

        options = OpenAIOptions.model_validate(option_kwargs)
        request_options = options.model_dump(exclude_none=True)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(request_options.get("extra_headers", {}))

        return OpenAIRequest(
            provider="openai",
            operation="responses.create",
            payload=payload,
            call_options=options,
            method="POST",
            url=OPENAI_RESPONSES_URL,
            headers=headers,
            params=request_options.get("extra_query"),
            timeout=request_options.get("timeout", self.client_timeout),
            extra_body=request_options.get("extra_body"),
        )

    def forward(self, argument):
        if self.id() != "neurosymbolic":
            msg = (
                "OpenAI engine is not configured. Please set a supported "
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

    def call_request(self, request: OpenAIRequest):
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
        return OpenAIResponse.model_validate(response.json())

    def parse_response(self, response: OpenAIResponse):
        if response.status != "completed":
            msg = f"OpenAI response status is {response.status!r}, expected 'completed'."
            if response.error:
                msg = f"{msg} Error: {response.error}"
            if response.incomplete_details:
                msg = f"{msg} Details: {response.incomplete_details}"
            raise ValueError(msg)

        metadata = {"raw_output": response, "thinking": self._extract_thinking(response)}
        metadata = self._process_function_calls(response, metadata)

        output = self._extract_output_text(response)
        if not output and "function_call" in metadata:
            output = [""]
        return output, metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        image_files = self._handle_image_content(str(argument.prop.processed_input))
        if image_files and not self.supports_vision():
            msg = f"Model {self.model} does not support vision input."
            raise ValueError(msg)

        system = render_chat_system_prompt(argument)
        user_text = self._build_user_text(argument, image_files)
        user_msg = self._create_user_message(user_text, image_files)
        system, user_msg = self._apply_self_prompt(
            argument, system, user_msg, user_text, image_files
        )

        role = "developer" if self.is_reasoning_model() else "system"
        argument.prop.prepared_input = [
            {"role": role, "content": system},
            user_msg,
        ]

    def _apply_self_prompt(self, argument, system, user_msg, user_text, image_files):
        prop = argument.prop
        if not (prop.instance._kwargs.get("self_prompt", False) or prop.self_prompt):
            return system, user_msg

        key = "developer" if self.is_reasoning_model() else "system"
        res = self.self_prompt({"user": user_text, key: system})
        if res is None:
            msg = "Self-prompting failed for OpenAIEngine."
            raise ValueError(msg)

        return res[key], self._create_user_message(res["user"], image_files)

    def _handle_image_content(self, content: str) -> list[str]:
        # NOTE: matches <<vision:...:>> markers embedding image references in the prompt.
        if "<<vision:" not in content:
            return []

        image_files = []
        for p in re.findall(r"<<vision:(.*?):>>", content):
            img_ = p.strip()
            if img_.startswith("http") or img_.startswith("data:image"):
                image_files.append(img_)
                continue
            max_frames_spacing = 50
            max_used_frames = 10
            if img_.startswith("frames:"):
                img_ = img_.replace("frames:", "")
                max_used_frames, img_ = img_.split(":")
                max_used_frames = int(max_used_frames)
                if max_used_frames < 1 or max_used_frames > max_frames_spacing:
                    msg = f"Invalid max_used_frames value: {max_used_frames}. Expected 1-{max_frames_spacing}"
                    raise ValueError(msg)
            buffer, ext = encode_media_frames(img_)
            if len(buffer) > 1:
                step = max(1, len(buffer) // max_frames_spacing)
                indices = list(range(0, len(buffer), step))[:max_used_frames]
                for i in indices:
                    image_files.append(f"data:image/{ext};base64,{buffer[i]}")
            elif len(buffer) == 1:
                image_files.append(f"data:image/{ext};base64,{buffer[0]}")
            else:
                logger.warning("No frames found or error in encoding frames")
        return image_files

    def _remove_vision_pattern(self, text: str) -> str:
        # NOTE: strips <<vision:...:>> markers from text after image extraction.
        return re.sub(r"<<vision:(.*?):>>", "", text)

    def _build_user_text(self, argument, image_files: list[str]) -> str:
        suffix = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = self._remove_vision_pattern(suffix)
        return suffix

    def _create_user_message(self, user_text: str, image_files: list[str]) -> dict:
        if image_files:
            images = [{"type": "input_image", "image_url": f} for f in image_files]
            return {"role": "user", "content": [*images, {"type": "input_text", "text": user_text}]}
        return {"role": "user", "content": user_text}

    def _prepare_raw_input(self, argument):
        value = argument.prop.processed_input
        if not value:
            msg = "A prompt instruction is required for OpenAIEngine when raw_input is enabled."
            raise ValueError(msg)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [{"role": "user", "content": str(value)}]

    def _collect_stream_response(self, request: OpenAIRequest, max_retries: int):
        adapter = OpenAIStreamAdapter()
        accumulator = EngineStreamAccumulator()
        completed_response = None

        for event in execute_engine_api_stream_events(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        ):
            delta = adapter.process_event(event)
            accumulator.add(delta)
            if accumulator.done:
                # NOTE: response.completed carries the full final Response object;
                # validating it directly is more faithful than reassembling deltas.
                if isinstance(delta.raw, dict) and "status" in delta.raw:
                    completed_response = delta.raw
                break

        if completed_response is not None:
            return OpenAIResponse.model_validate(completed_response)

        # Fallback: the stream ended without a terminal event; reassemble partial output.
        message_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": accumulator.text}],
        }
        items = [message_item]
        if accumulator.thinking:
            items.insert(
                0,
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": accumulator.thinking}],
                },
            )

        return OpenAIResponse.model_validate(
            {
                "status": "completed",
                "output": items,
                "usage": accumulator.usage,
            }
        )

    def _apply_cache_breakpoints(self, messages: list[dict], model_spec, model: str) -> list[dict]:
        """Translate ``symai:cache_breakpoint`` markers into explicit cache blocks.

        Markers on models without explicit caching raise (a misplaced marker signals a
        wrong model choice, not a no-op). Unmarked messages pass through unchanged."""
        marker_count = sum(
            str(message.get("content", "")).count(CACHE_BREAKPOINT) for message in messages
        )
        if marker_count == 0:
            return messages
        if not model_spec.explicit_cache:
            msg = f"Explicit cache breakpoints are not supported by {model}; use a GPT-5.6 model."
            raise ValueError(msg)

        prepared = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                if CACHE_BREAKPOINT in content:
                    content = build_cache_breakpoint_blocks(content)
                prepared.append({**message, "content": content})
                continue
            blocks = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "input_text"
                    and CACHE_BREAKPOINT in block.get("text", "")
                ):
                    blocks.extend(build_cache_breakpoint_blocks(block["text"]))
                else:
                    blocks.append(block)
            prepared.append({**message, "content": blocks})
        return prepared

    @staticmethod
    def _normalize_response_format(response_format):
        # NOTE: the chat-completions nested form {"type": "json_schema", "json_schema": {...}}
        # flattens to {"type": "json_schema", "name": ..., "schema": ...} in the Responses
        # API text.format field.
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            nested = response_format.get("json_schema")
            if isinstance(nested, dict):
                return {"type": "json_schema", **nested}
        return response_format

    def _convert_tools(self, tools: list) -> list:
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                converted.append(
                    {
                        "type": "function",
                        "name": tool.get("name") or tool.get("function", {}).get("name"),
                        "description": tool.get("description")
                        or tool.get("function", {}).get("description"),
                        "parameters": tool.get("parameters")
                        or tool.get("function", {}).get("parameters"),
                    }
                )
            else:
                converted.append(tool)
        return converted

    def _extract_output_text(self, response: OpenAIResponse) -> list[str]:
        outputs = []
        for item in response.output:
            if item.type == "message" and item.content:
                for content in item.content:
                    outputs.append(content.text)
        return outputs

    def _extract_thinking(self, response: OpenAIResponse) -> str | None:
        if not self.is_reasoning_model():
            return None
        for item in response.output:
            if item.type == "reasoning" and item.summary:
                texts = [s.text for s in item.summary if s.text]
                if texts:
                    return "\n".join(texts)
        return None

    def _process_function_calls(self, response: OpenAIResponse, metadata: dict) -> dict:
        for item in response.output:
            if item.type == "function_call":
                try:
                    args_dict = json.loads(item.arguments)
                except (json.JSONDecodeError, TypeError):
                    args_dict = {}
                metadata["function_call"] = {
                    "name": item.name,
                    "arguments": args_dict,
                    "call_id": item.call_id,
                }
                break
        return metadata
