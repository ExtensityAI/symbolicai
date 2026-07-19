from __future__ import annotations

import json
import logging
from copy import deepcopy

import httpx

from symai.backend.base import Engine
from symai.backend.engines.neurosymbolic._prompts import render_chat_system_prompt
from symai.backend.engines.neurosymbolic.vllm.models import (
    VLLMModelsResponse,
    VLLMOptions,
    VLLMPayload,
    VLLMRequest,
    VLLMResponse,
    VLLMTokenizePayload,
    VLLMTokenizeRequest,
    VLLMTokenizeResponse,
    vllm_model_spec_for,
)
from symai.backend.engines.neurosymbolic.vllm.stream import VLLMStreamAdapter
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


class VLLMEngine(Engine):
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
        # NOTE: vLLM serves without auth by default but supports --api-key; honor it
        # when the caller passes one (Bearer header).
        self.api_key = api_key
        if model is not None:
            self.config["NEUROSYMBOLIC_ENGINE_MODEL"] = model
        self.model = self.config["NEUROSYMBOLIC_ENGINE_MODEL"]
        if self.id() != "neurosymbolic":
            return
        if not SYMSERVER_CONFIG.get("online"):
            msg = (
                "You are using the vLLM engine, but the server endpoint is not started. "
                "Please start the server with `symserver --model <hf-repo-id> [--args]` or run "
                "`symserver --help` to see the available options for this engine."
            )
            raise ValueError(msg)
        host = SYMSERVER_CONFIG.get("--host") or "localhost"
        port = SYMSERVER_CONFIG.get("--port") or 8000
        self.server_endpoint = f"http://{host}:{port}"
        self.tokenizer = None
        self.transport_client = None
        self.server_model = self._server_model_id()
        self.max_context_tokens = self._server_context_tokens()
        self.max_response_tokens = self.max_context_tokens

    def id(self) -> str:
        if self.model and self.model.startswith("vllm"):
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def api_max_context_tokens(self) -> int:
        return vllm_model_spec_for(self.model).context_tokens

    def _server_headers(self) -> dict[str, str]:
        # NOTE: symserver runs without auth by default but supports --api-key; send the
        # Bearer token on every server request (discovery, tokenize, chat) so keyed
        # deployments don't 401 on the helper calls.
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _server_get(self, path: str) -> dict:
        if self.transport_client is not None:
            return self.transport_client.get(
                f"{self.server_endpoint}{path}", headers=self._server_headers()
            ).json()

        return httpx.get(
            f"{self.server_endpoint}{path}", headers=self._server_headers(), timeout=10.0
        ).json()

    def _server_model_id(self) -> str | None:
        try:
            models = VLLMModelsResponse.model_validate(self._server_get("/v1/models"))
        except Exception as e:
            logger.warning("Failed to query vLLM /v1/models: %s", e)
            return None
        return models.data[0].id if models.data else None

    def _server_context_tokens(self) -> int:
        try:
            models = VLLMModelsResponse.model_validate(self._server_get("/v1/models"))
            if models.data and models.data[0].max_model_len:
                return models.data[0].max_model_len
        except Exception as e:
            logger.warning("Failed to query vLLM /v1/models: %s", e)
        return 4096

    def compute_required_tokens(self, messages: list[dict]) -> int:
        # NOTE: vLLM has no template-rendering endpoint, so this tokenizes the message
        # text plus a per-message chat-template overhead (6, calibrated against Qwen3's
        # template on llama.cpp's exact renderer; model-dependent, an estimate).
        num_tokens = 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                content = json.dumps(content)
            num_tokens += self._tokenize(content) + 6
        return num_tokens

    def compute_remaining_tokens(self, prompts: list[dict]) -> int:
        val = self.compute_required_tokens(prompts)
        return max(self.max_context_tokens - val, 0)

    def _tokenize(self, text: str) -> int:
        request = VLLMTokenizeRequest(
            provider="vllm",
            operation="tokenize",
            payload=VLLMTokenizePayload(model=self.server_model or "", prompt=text),
            method="POST",
            url=f"{self.server_endpoint}/tokenize",
            headers=self._server_headers(),
        )
        response = execute_engine_api_request(request, client=self.transport_client)
        return VLLMTokenizeResponse.model_validate(response.json()).count

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord:
        usage = metadata["raw_output"].usage

        return EngineUsageRecord(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def build_request(self, argument) -> VLLMRequest:
        allowed_request_kwargs = set(VLLMPayload.model_fields).union(VLLMOptions.model_fields)
        payload_kwargs = self.collect_request_kwargs(argument, allowed_request_kwargs)
        option_kwargs = {
            key: payload_kwargs.pop(key)
            for key in VLLMOptions.model_fields
            if key in payload_kwargs
        }

        payload_kwargs["messages"] = strip_cache_breakpoints_from_messages(
            argument.prop.prepared_input
        )
        # NOTE: vLLM rejects empty stop strings outright; omit stop when unset.
        if not payload_kwargs.get("stop"):
            payload_kwargs["stop"] = None
        if self.server_model:
            payload_kwargs.setdefault("model", self.server_model)

        if payload_kwargs.get("stream"):
            # NOTE: usage is required on VLLMResponse (MetadataTracker reads it), and
            # streams only carry usage in the final chunk when include_usage is set.
            payload_kwargs.setdefault("stream_options", {"include_usage": True})

        payload = VLLMPayload.model_validate(payload_kwargs)
        options = VLLMOptions.model_validate(option_kwargs)
        request_options = options.model_dump(exclude_none=True)

        headers = self._server_headers()
        headers.update(request_options.get("extra_headers", {}))

        return VLLMRequest(
            provider="vllm",
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
                "vLLM engine is not configured. Please set NEUROSYMBOLIC_ENGINE_MODEL "
                "to a vllm* value and start the server with `symserver --model <hf-repo-id>`."
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

    def call_request(self, request: VLLMRequest):
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
        return VLLMResponse.model_validate(response.json())

    def parse_response(self, response: VLLMResponse):
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
                msg = "Self-prompting failed for VLLMEngine."
                raise ValueError(msg)

            user_prompt = {"role": "user", "content": res["user"]}
            system = res["system"]

        return system, user_prompt

    def _collect_stream_response(self, request: VLLMRequest, max_retries: int):
        adapter = VLLMStreamAdapter()
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

        return VLLMResponse.model_validate(
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
            msg = "A prompt instruction is required for VLLMEngine when raw_input is enabled."
            raise ValueError(msg)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [{"role": "user", "content": str(value)}]

    def _process_tool_calls(self, response: VLLMResponse, metadata: dict) -> dict:
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
