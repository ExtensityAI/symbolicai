from __future__ import annotations

import json
import logging
from copy import deepcopy

import httpx

from symai.backend.base import Engine
from symai.backend.engines.neurosymbolic._prompts import render_chat_system_prompt
from symai.backend.engines.neurosymbolic.llama_cpp.models import (
    LlamaCppApplyTemplatePayload,
    LlamaCppApplyTemplateRequest,
    LlamaCppApplyTemplateResponse,
    LlamaCppDetokenizePayload,
    LlamaCppDetokenizeRequest,
    LlamaCppDetokenizeResponse,
    LlamaCppOptions,
    LlamaCppPayload,
    LlamaCppRequest,
    LlamaCppResponse,
    LlamaCppTokenizePayload,
    LlamaCppTokenizeRequest,
    LlamaCppTokenizeResponse,
    llamacpp_model_spec_for,
)
from symai.backend.engines.neurosymbolic.llama_cpp.stream import LlamaCppStreamAdapter
from symai.backend.settings import SYMAI_CONFIG, SYMSERVER_CONFIG
from symai.backend.streaming import EngineStreamAccumulator
from symai.backend.transport import (
    DEFAULT_RETRIES,
    execute_engine_api_request,
    execute_engine_api_stream_events,
)
from symai.backend.usage import EngineUsageRecord
from symai.prompts import strip_cache_breakpoints_from_messages

logger = logging.getLogger(__name__)


class LlamaCppEngine(Engine):
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
        # NOTE: llama-server runs without auth by default but supports --api-key; honor
        # it when the caller passes one (Bearer header).
        self.api_key = api_key
        if model is not None:
            self.config["NEUROSYMBOLIC_ENGINE_MODEL"] = model
        self.model = self.config["NEUROSYMBOLIC_ENGINE_MODEL"]
        if self.id() != "neurosymbolic":
            return
        if not SYMSERVER_CONFIG.get("online"):
            msg = (
                "You are using the llama.cpp engine, but the server endpoint is not started. "
                "Please start the server with `symserver [--args]` or run `symserver --help` "
                "to see the available options for this engine."
            )
            raise ValueError(msg)
        self.server_endpoint = (
            f"http://{SYMSERVER_CONFIG.get('--host')}:{SYMSERVER_CONFIG.get('--port')}"
        )
        self.tokenizer = None
        self.transport_client = None
        self.max_context_tokens = self._server_context_tokens()
        self.max_response_tokens = self.max_context_tokens

    def id(self) -> str:
        if self.model and self.model.startswith("llama"):
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def api_max_context_tokens(self) -> int:
        return llamacpp_model_spec_for(self.model).context_tokens

    def _server_context_tokens(self) -> int:
        try:
            props = self._server_get("/props")
        except Exception as e:
            msg = (
                f"Failed to query the llama.cpp server at {self.server_endpoint}/props "
                "during engine initialization. Ensure the server is running "
                f"(`symserver [--args]`) and reachable. Caused by: {e}"
            )
            raise ValueError(msg) from e
        n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        return n_ctx or 4096

    def _server_headers(self) -> dict[str, str]:
        # NOTE: llama-server runs without auth by default but supports --api-key; send
        # the Bearer token on every server request (discovery, template, tokenize,
        # detokenize, chat) so keyed deployments don't 401 on the helper calls.
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _server_get(self, path: str) -> dict:
        if self.transport_client is not None:
            response = self.transport_client.get(
                f"{self.server_endpoint}{path}", headers=self._server_headers()
            )
            return response.json()

        response = httpx.get(
            f"{self.server_endpoint}{path}", headers=self._server_headers(), timeout=10.0
        )
        return response.json()

    def compute_required_tokens(self, messages: list[dict]) -> int:
        # NOTE: exact count — the server renders the loaded model's chat template and
        # tokenizes the result, so framing overhead matches generation precisely.
        prompt = self._apply_template(messages)
        return len(self._tokenize(prompt))

    def compute_remaining_tokens(self, prompts: list[dict]) -> int:
        val = self.compute_required_tokens(prompts)
        return max(self.max_context_tokens - val, 0)

    def _apply_template(self, messages: list[dict]) -> str:
        request = LlamaCppApplyTemplateRequest(
            provider="llamacpp",
            operation="apply_template",
            payload=LlamaCppApplyTemplatePayload(messages=messages),
            method="POST",
            url=f"{self.server_endpoint}/apply-template",
            headers=self._server_headers(),
        )
        response = execute_engine_api_request(request, client=self.transport_client)
        return LlamaCppApplyTemplateResponse.model_validate(response.json()).prompt

    def _tokenize(self, text: str) -> list[int]:
        request = LlamaCppTokenizeRequest(
            provider="llamacpp",
            operation="tokenize",
            payload=LlamaCppTokenizePayload(content=text),
            method="POST",
            url=f"{self.server_endpoint}/tokenize",
            headers=self._server_headers(),
        )
        response = execute_engine_api_request(request, client=self.transport_client)
        return LlamaCppTokenizeResponse.model_validate(response.json()).tokens

    def _detokenize(self, tokens: list[int]) -> str:
        request = LlamaCppDetokenizeRequest(
            provider="llamacpp",
            operation="detokenize",
            payload=LlamaCppDetokenizePayload(tokens=tokens),
            method="POST",
            url=f"{self.server_endpoint}/detokenize",
            headers=self._server_headers(),
        )
        response = execute_engine_api_request(request, client=self.transport_client)
        return LlamaCppDetokenizeResponse.model_validate(response.json()).content

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord:
        usage = metadata["raw_output"].usage
        prompt_details = usage.prompt_tokens_details

        return EngineUsageRecord(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            prompt_breakdown={
                "cached_tokens": (prompt_details.cached_tokens or 0) if prompt_details else 0,
            },
        )

    def build_request(self, argument) -> LlamaCppRequest:
        allowed_request_kwargs = set(LlamaCppPayload.model_fields).union(
            LlamaCppOptions.model_fields
        )
        payload_kwargs = self.collect_request_kwargs(argument, allowed_request_kwargs)
        option_kwargs = {
            key: payload_kwargs.pop(key)
            for key in LlamaCppOptions.model_fields
            if key in payload_kwargs
        }

        payload_kwargs["messages"] = strip_cache_breakpoints_from_messages(
            argument.prop.prepared_input
        )
        # NOTE: llama.cpp server defaults (temperature 0.8, repeat_penalty 1.1) differ
        # from what this engine historically ships; keep the engine's own defaults.
        payload_kwargs.setdefault("temperature", 0.6)
        payload_kwargs.setdefault("frequency_penalty", 0)
        payload_kwargs.setdefault("presence_penalty", 0)
        payload_kwargs.setdefault("top_p", 0.95)
        payload_kwargs.setdefault("min_p", 0.05)
        payload_kwargs.setdefault("top_k", 40)
        payload_kwargs.setdefault("repeat_penalty", 1)
        # NOTE: core decorators default stop="" (meaning unset); omit it when unset.
        if not payload_kwargs.get("stop"):
            payload_kwargs["stop"] = None

        server_model = SYMSERVER_CONFIG.get("-m") or SYMSERVER_CONFIG.get("--model")
        if server_model:
            payload_kwargs.setdefault("model", server_model)

        if payload_kwargs.get("stream"):
            # NOTE: usage is required on LlamaCppResponse (MetadataTracker reads it),
            # and streams only carry usage in the final chunk when include_usage is set.
            payload_kwargs.setdefault("stream_options", {"include_usage": True})

        payload = LlamaCppPayload.model_validate(payload_kwargs)
        options = LlamaCppOptions.model_validate(option_kwargs)
        request_options = options.model_dump(exclude_none=True)

        headers = self._server_headers()
        headers.update(request_options.get("extra_headers", {}))

        return LlamaCppRequest(
            provider="llamacpp",
            operation="chat.completions.create",
            payload=payload,
            call_options=options,
            method="POST",
            url=f"{self.server_endpoint}/v1/chat/completions",
            headers=headers,
            params=request_options.get("extra_query"),
            timeout=request_options.get("timeout", self.client_timeout),
            extra_body=request_options.get("extra_body"),
        )

    def forward(self, argument):
        if self.id() != "neurosymbolic":
            msg = (
                "llama.cpp engine is not configured. Please set NEUROSYMBOLIC_ENGINE_MODEL "
                "to a llama* value and start the server with `symserver [--args]`."
            )
            raise ValueError(msg)

        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def call_request(self, request: LlamaCppRequest):
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
        return LlamaCppResponse.model_validate(response.json())

    def parse_response(self, response: LlamaCppResponse):
        metadata: dict = {"raw_output": response}
        metadata = self._process_tool_calls(response, metadata)

        message = response.choices[0].message
        if message.reasoning_content:
            metadata["thinking"] = message.reasoning_content

        return [message.content or ""], metadata

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
                msg = "Self-prompting failed for LlamaCppEngine."
                raise ValueError(msg)

            user_prompt = {"role": "user", "content": res["user"]}
            system = res["system"]

        return system, user_prompt

    def _collect_stream_response(self, request: LlamaCppRequest, max_retries: int):
        adapter = LlamaCppStreamAdapter()
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

        return LlamaCppResponse.model_validate(
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
            msg = "A prompt instruction is required for LlamaCppEngine when raw_input is enabled."
            raise ValueError(msg)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [{"role": "user", "content": str(value)}]

    def _process_tool_calls(self, response: LlamaCppResponse, metadata: dict) -> dict:
        message = response.choices[0].message
        if not message.tool_calls:
            return metadata

        for tool_call in message.tool_calls:
            function = tool_call.function or {}
            if "function_call" in metadata:
                logger.warning(
                    "Multiple function calls detected in the response but only the first one will be processed."
                )
                break
            arguments = function.get("arguments")
            try:
                args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments or {}
            except json.JSONDecodeError:
                args_dict = {}
            metadata["function_call"] = {
                "name": function.get("name"),
                "arguments": args_dict,
            }

        return metadata
