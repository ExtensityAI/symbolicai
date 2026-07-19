from __future__ import annotations

import base64
import json
import logging
import re
from copy import deepcopy

import httpx

from symai.backend.base import Engine
from symai.backend.engines.neurosymbolic._prompts import render_chat_system_prompt
from symai.backend.engines.neurosymbolic.google.models import (
    SUPPORTED_GOOGLE_MODELS,
    GoogleCountTokensPayload,
    GoogleCountTokensRequest,
    GoogleCountTokensResponse,
    GoogleGenerationConfig,
    GoogleOptions,
    GooglePayload,
    GoogleRequest,
    GoogleResponse,
    GoogleSystemInstruction,
    GoogleTool,
    google_model_spec_for,
    google_strip_prefix,
)
from symai.backend.engines.neurosymbolic.google.stream import GoogleStreamAdapter
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.streaming import EngineStreamAccumulator
from symai.backend.transport import (
    DEFAULT_RETRIES,
    execute_engine_api_request,
    execute_engine_api_stream_events,
)
from symai.backend.usage import EngineUsageRecord
from symai.prompts import strip_cache_breakpoints_from_messages
from symai.utils import encode_media_frames

logger = logging.getLogger(__name__)

GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleEngine(Engine):
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
        if self.model in SUPPORTED_GOOGLE_MODELS and self.api_key:
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def api_max_context_tokens(self) -> int:
        return google_model_spec_for(self.model).context_tokens

    def api_max_response_tokens(self) -> int:
        return google_model_spec_for(self.model).response_tokens

    def is_reasoning_model(self) -> bool:
        return google_model_spec_for(self.model).reasoning

    def compute_required_tokens(self, messages: list[dict]) -> int:
        contents = self._build_count_contents(messages)
        if not contents:
            return 0

        payload = {"contents": contents}
        model = google_strip_prefix(self.model)
        request = GoogleCountTokensRequest(
            provider="google",
            operation="models.count_tokens",
            payload=GoogleCountTokensPayload.model_validate(payload),
            method="POST",
            url=f"{GOOGLE_API_BASE}/{model}:countTokens",
            headers=self._auth_headers(),
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=self.client_max_retries
            if self.client_max_retries is not None
            else DEFAULT_RETRIES,
        )
        return GoogleCountTokensResponse.model_validate(response.json()).total_tokens

    def compute_remaining_tokens(self, prompts: list[dict]) -> int:
        val = self.compute_required_tokens(prompts)
        return max(self.max_context_tokens - val, 0)

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord:
        usage = metadata["raw_output"].usage_metadata
        thoughts = usage.thoughts_token_count or 0
        candidates = usage.candidates_token_count or 0

        return EngineUsageRecord(
            prompt_tokens=usage.prompt_token_count,
            # NOTE: Gemini bills output including thinking tokens.
            completion_tokens=candidates + thoughts,
            total_tokens=usage.total_token_count,
            prompt_breakdown={
                "cached_tokens": usage.cached_content_token_count or 0,
            },
            completion_breakdown={
                "reasoning_tokens": thoughts,
            },
        )

    def build_request(self, argument) -> GoogleRequest:
        allowed_request_kwargs = set(GooglePayload.model_fields).union(
            GoogleOptions.model_fields
        ) | {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "thinking",
            "response_format",
            "stream",
        }
        payload_kwargs = self.collect_request_kwargs(argument, allowed_request_kwargs)
        option_kwargs = {
            key: payload_kwargs.pop(key)
            for key in GoogleOptions.model_fields
            if key in payload_kwargs
        }

        model = google_strip_prefix(payload_kwargs.get("model", self.model))
        google_model_spec_for(model)
        system, contents = self._build_wire_contents(
            strip_cache_breakpoints_from_messages(argument.prop.prepared_input)
        )
        stream = bool(payload_kwargs.pop("stream", False))

        generation_config: dict = {}
        if "max_tokens" in payload_kwargs:
            generation_config["max_output_tokens"] = payload_kwargs.pop("max_tokens")
        for key in ("temperature", "top_p", "top_k"):
            if key in payload_kwargs:
                generation_config[key] = payload_kwargs.pop(key)
        generation_config.setdefault("temperature", 1.0)
        generation_config.setdefault("top_p", 0.95)
        generation_config.setdefault("top_k", 40)

        stop = payload_kwargs.pop("stop", None)
        if stop:
            generation_config["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        response_format = payload_kwargs.pop("response_format", None)
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            generation_config["response_mime_type"] = "application/json"

        thinking = payload_kwargs.pop("thinking", None)
        if isinstance(thinking, dict) and self.is_reasoning_model():
            thinking_config = {"includeThoughts": True}
            if "thinking_level" in thinking:
                thinking_config["thinkingLevel"] = thinking["thinking_level"]
            else:
                thinking_config["thinkingBudget"] = thinking.get("thinking_budget", 1024)
            generation_config["thinking_config"] = thinking_config

        payload: dict = {"contents": contents}
        if system:
            payload["system_instruction"] = GoogleSystemInstruction(parts=[{"text": system}])
        if generation_config:
            payload["generation_config"] = GoogleGenerationConfig.model_validate(generation_config)
        if payload_kwargs.get("tools"):
            payload["tools"] = self._convert_tools(payload_kwargs.pop("tools"))
        tool_config = payload_kwargs.pop("tool_config", None)
        if tool_config is not None:
            payload["tool_config"] = tool_config

        payload = GooglePayload.model_validate(payload)
        options = GoogleOptions.model_validate(option_kwargs)
        request_options = options.model_dump(exclude_none=True)

        headers = self._auth_headers()
        headers.update(request_options.get("extra_headers", {}))

        if stream:
            # NOTE: Gemini streams over a sibling endpoint with SSE framing, not a
            # payload flag: {model}:streamGenerateContent?alt=sse
            params = dict(request_options.get("extra_query") or {})
            params["alt"] = "sse"
            return GoogleRequest(
                provider="google",
                operation="models.stream_generate_content",
                payload=payload,
                call_options=options,
                method="POST",
                url=f"{GOOGLE_API_BASE}/{model}:streamGenerateContent",
                headers=headers,
                params=params,
                timeout=request_options.get("timeout", self.client_timeout),
                extra_body=request_options.get("extra_body"),
            )

        return GoogleRequest(
            provider="google",
            operation="models.generate_content",
            payload=payload,
            call_options=options,
            method="POST",
            url=f"{GOOGLE_API_BASE}/{model}:generateContent",
            headers=headers,
            params=request_options.get("extra_query"),
            timeout=request_options.get("timeout", self.client_timeout),
            extra_body=request_options.get("extra_body"),
        )

    def forward(self, argument):
        if self.id() != "neurosymbolic":
            msg = (
                "Google engine is not configured. Please set a supported "
                "NEUROSYMBOLIC_ENGINE_MODEL and NEUROSYMBOLIC_ENGINE_API_KEY."
            )
            raise ValueError(msg)

        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response, argument)

    def call_request(self, request: GoogleRequest):
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        if request.operation == "models.stream_generate_content":
            return self._collect_stream_response(request, max_retries)
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        return GoogleResponse.model_validate(response.json())

    def parse_response(self, response: GoogleResponse, argument=None):
        metadata: dict = {"raw_output": response}
        metadata = self._process_function_calls(response, metadata)

        outputs = []
        thinking_parts = []
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.text is None:
                    continue
                if part.thought:
                    thinking_parts.append(part.text)
                else:
                    outputs.append(part.text)

        metadata["thinking"] = "\n".join(thinking_parts) or None

        text_output = "".join(outputs)
        if argument is not None and getattr(argument.prop, "response_format", None):
            # NOTE: Gemini wraps JSON output in markdown fences.
            text_output = text_output.replace("```json", "").replace("```", "")

        return [text_output], metadata

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
                msg = "Self-prompting failed for GoogleEngine."
                raise ValueError(msg)

            user_prompt = {"role": "user", "content": res["user"]}
            system = res["system"]

        return system, user_prompt

    def _auth_headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _build_count_contents(self, messages: list[dict]) -> list[dict]:
        """Contents for countTokens: roles preserved (system is a valid count role),
        assistant mapped to model, text wrapped in parts."""
        contents = []
        for message in messages:
            role = message.get("role")
            wire_role = "model" if role in ("assistant", "model") else role
            parts = self._build_parts(message.get("content"))
            if parts:
                contents.append({"role": wire_role, "parts": parts})
        return contents

    def _build_wire_contents(self, prepared_input) -> tuple[str | None, list[dict]]:
        """Split uniform prepared input into Gemini's systemInstruction + contents."""
        system = None
        contents = []
        for message in prepared_input:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                system = content if isinstance(content, str) else json.dumps(content)
                continue
            wire_role = "model" if role in ("assistant", "model") else "user"
            parts = self._build_parts(content)
            if parts:
                contents.append({"role": wire_role, "parts": parts})
        return system, contents

    def _build_parts(self, content) -> list[dict]:
        if not isinstance(content, str):
            return content if isinstance(content, list) else [content]

        image_parts = self._handle_image_content(content)
        text = self._remove_media_patterns(content)
        parts = [*image_parts]
        if text:
            parts.append({"text": text})
        if not parts:
            # NOTE: Gemini rejects empty content lists.
            parts = [{"text": "N/A"}]
        return parts

    def _handle_image_content(self, content: str) -> list[dict]:
        # NOTE: matches <<vision:...:>> markers embedding image references in the prompt.
        if "<<vision:" not in content:
            return []

        image_parts = []
        for p in re.findall(r"<<vision:(.*?):>>", content):
            img_ = p.strip()
            if img_.startswith("data:image"):
                header, encoded = img_.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                image_parts.append({"inline_data": {"mime_type": mime_type, "data": encoded}})
                continue
            if img_.startswith(("http://", "https://")):
                response = httpx.get(img_, timeout=10, follow_redirects=True)
                response.raise_for_status()
                mime_type = response.headers.get("Content-Type", "application/octet-stream")
                image_parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(response.content).decode(),
                        }
                    }
                )
                continue
            max_used_frames = 10
            if img_.startswith("frames:"):
                img_ = img_.replace("frames:", "")
                max_used_frames, img_ = img_.split(":")
                max_used_frames = int(max_used_frames)
            buffer, ext = encode_media_frames(img_)
            if not buffer:
                logger.warning("No frames found or error in encoding frames")
                continue
            mime_type = "image/jpeg" if ext and ext.lower() in ("jpg", "jpeg") else f"image/{ext}"
            step = max(1, len(buffer) // 50)
            indices = list(range(0, len(buffer), step))[:max_used_frames]
            for i in indices:
                image_parts.append({"inline_data": {"mime_type": mime_type, "data": buffer[i]}})
        return image_parts

    def _remove_media_patterns(self, text: str) -> str:
        # NOTE: strips <<vision:...:>> markers after image extraction. Video, audio,
        # and document markers require Gemini's Files API upload flow, which the
        # raw-REST engine does not implement — refuse loudly instead of silently
        # dropping the user's media.
        unsupported = sorted(set(re.findall(r"<<(video|audio|document):(.*?):>>", text)))
        if unsupported:
            kinds = ", ".join(f"<<{kind}:...:>>" for kind, _ in unsupported)
            msg = (
                f"GoogleEngine does not support {kinds} media markers: the Gemini Files "
                "API upload flow used by the legacy engine is not implemented over raw "
                "REST. Remove the marker or use <<vision:...:>> image markers."
            )
            raise NotImplementedError(msg)
        return re.sub(r"<<vision:(.*?):>>", "", text)

    def _convert_tools(self, tools) -> list[GoogleTool]:
        # NOTE: only dict function declarations are supported over the wire; local
        # callables and SDK-side automatic function calling are not portable to raw HTTP.
        declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                declarations.append(
                    tool.get("function") or tool.get("function_declaration") or tool
                )
            elif "name" in tool:
                declarations.append(tool)
            else:
                logger.warning("Ignoring invalid tool format: %s", tool)
        if not declarations:
            return None
        return [GoogleTool(function_declarations=declarations)]

    def _collect_stream_response(self, request: GoogleRequest, max_retries: int):
        adapter = GoogleStreamAdapter()
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

        parts = []
        if accumulator.thinking:
            parts.append({"text": accumulator.thinking, "thought": True})
        parts.append({"text": accumulator.text})

        return GoogleResponse.model_validate(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": parts},
                        "finishReason": accumulator.finish_reason,
                    }
                ],
                "usageMetadata": accumulator.usage,
            }
        )

    def _prepare_raw_input(self, argument):
        value = argument.prop.processed_input
        if not value:
            msg = "A prompt instruction is required for GoogleEngine when raw_input is enabled."
            raise ValueError(msg)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [{"role": "user", "content": str(value)}]

    def _process_function_calls(self, response: GoogleResponse, metadata: dict) -> dict:
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return metadata

        for part in candidate.content.parts:
            if part.function_call:
                if "function_call" in metadata:
                    logger.warning(
                        "Multiple function calls detected in the response but only the first one will be processed."
                    )
                    break
                metadata["function_call"] = {
                    "name": part.function_call.get("name"),
                    "arguments": part.function_call.get("args") or {},
                }
        return metadata
