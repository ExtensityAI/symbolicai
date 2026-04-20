import asyncio
import json
import logging
from copy import deepcopy
from typing import Any, ClassVar

import aiohttp
import nest_asyncio

from ....core import Argument
from ....core_ext import retry
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG, SYMSERVER_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class VLLMEngine(Engine):
    _retry_params: ClassVar[dict[str, Any]] = {
        "tries": 5,
        "delay": 2,
        "max_delay": 60,
        "backoff": 2,
        "jitter": (1, 5),
        "graceful": True,
    }
    _timeout_params: ClassVar[dict[str, Any]] = {
        "read": None,
        "connect": None,
    }

    def __init__(
        self,
        model: str | None = None,
        retry_params: dict = _retry_params,
        timeout_params: dict = _timeout_params,
    ):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        # NOTE: allow EngineRepository.register to inject a model and flip id() at runtime
        if model is not None:
            self.config["NEUROSYMBOLIC_ENGINE_MODEL"] = model
        if self.id() != "neurosymbolic":
            return
        if not SYMSERVER_CONFIG.get("online"):
            UserMessage(
                "You are using the vLLM engine, but the server endpoint is not started. "
                "Please start the server with `symserver --model <hf-repo-id> [--args]` or run "
                "`symserver --help` to see the available options for this engine.",
                raise_with=ValueError,
            )
        host = SYMSERVER_CONFIG.get("--host") or "localhost"
        port = SYMSERVER_CONFIG.get("--port") or 8000
        self.server_endpoint = f"http://{host}:{port}"
        self.timeout_params = self._validate_timeout_params(timeout_params)
        self.retry_params = self._validate_retry_params(retry_params)
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config.get("NEUROSYMBOLIC_ENGINE_MODEL") and self.config.get(
            "NEUROSYMBOLIC_ENGINE_MODEL"
        ).startswith("vllm"):
            return "neurosymbolic"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "NEUROSYMBOLIC_ENGINE_MODEL" in kwargs:
            self.model = kwargs["NEUROSYMBOLIC_ENGINE_MODEL"]
        if "seed" in kwargs:
            self.seed = kwargs["seed"]
        if "except_remedy" in kwargs:
            self.except_remedy = kwargs["except_remedy"]

    def compute_required_tokens(self, _messages) -> int:
        # TODO: wire to vLLM's /tokenize endpoint once a VLLMTokenizer is added
        UserMessage("Not implemented for vLLM!", raise_with=NotImplementedError)

    def compute_remaining_tokens(self, _prompts: list) -> int:
        # TODO: wire to vLLM's /tokenize once VLLMTokenizer and max_model_len lookup exist
        UserMessage("Not implemented for vLLM!", raise_with=NotImplementedError)

    def _validate_timeout_params(self, timeout_params):
        assert all(key in timeout_params for key in ["read", "connect"]), (
            "Available keys: ['read', 'connect']"
        )
        return timeout_params

    def _validate_retry_params(self, retry_params):
        assert all(
            key in retry_params
            for key in ["tries", "delay", "max_delay", "backoff", "jitter", "graceful"]
        ), "Available keys: ['tries', 'delay', 'max_delay', 'backoff', 'jitter', 'graceful']"
        return retry_params

    @staticmethod
    def _get_event_loop() -> asyncio.AbstractEventLoop:
        """Gets or creates an event loop."""
        try:
            current_loop = asyncio.get_event_loop()
            if current_loop.is_closed():
                UserMessage("Event loop is closed.", raise_with=RuntimeError)
            return current_loop
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop

    def _prepare_request_payload(self, argument: Argument) -> dict:
        """Prepares the OpenAI-compatible chat/completions payload for vLLM."""
        kwargs = argument.kwargs
        # NOTE: only send fields the caller actually set — vLLM-specific sentinels
        # (e.g. top_k=-1) get rejected by stricter OpenAI-compatible servers, and
        # empty string for `stop` is rejected outright by vLLM.
        payload = {"messages": argument.prop.prepared_input}
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "max_tokens",
            "logprobs",
            "logit_bias",
            "response_format",
        ):
            value = kwargs.get(key)
            if value is None:
                continue
            if isinstance(value, (str, list, tuple)) and len(value) == 0:
                continue
            payload[key] = value

        model = SYMSERVER_CONFIG.get("--served-model-name") or SYMSERVER_CONFIG.get("--model")
        if model:
            payload["model"] = model

        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        # NOTE: extra_body lets users pass vLLM-specific knobs (guided_decoding, min_p,
        # repetition_penalty, chat_template_kwargs, ...) without growing this signature
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        return payload

    async def _arequest(self, payload: dict) -> dict:
        """Makes an async HTTP request to the vLLM server."""

        @retry(**self.retry_params)
        async def _make_request():
            timeout = aiohttp.ClientTimeout(
                sock_connect=self.timeout_params["connect"], sock_read=self.timeout_params["read"]
            )
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(f"{self.server_endpoint}/v1/chat/completions", json=payload) as res,
            ):
                if res.status != 200:
                    UserMessage(
                        f"Request failed with status code: {res.status}", raise_with=ValueError
                    )
                return await res.json()

        return await _make_request()

    @staticmethod
    def _extract_thinking(response):
        """Extract reasoning traces from vLLM responses (when --reasoning-parser is set)."""
        if not isinstance(response, dict):
            return None
        choices = response.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return None
        for choice in choices:
            if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                return choice["message"].get("reasoning_content")
        return None

    def forward(self, argument):
        payload = self._prepare_request_payload(argument)

        nest_asyncio.apply()
        loop = self._get_event_loop()

        try:
            res = loop.run_until_complete(self._arequest(payload))
        except Exception as e:
            UserMessage(f"Error during generation. Caused by: {e}", raise_with=ValueError)

        # NOTE: graceful retry returns None on repeated failure; surface that as a
        # real error instead of letting the unpack below crash with a cryptic TypeError.
        if res is None:
            UserMessage(
                "vLLM request returned no response after retries; check the server log.",
                raise_with=ValueError,
            )

        metadata = {"raw_output": res}

        if payload.get("tools"):
            metadata = self._process_tool_calls(res, metadata)

        thinking = self._extract_thinking(res)
        if thinking:
            metadata["thinking"] = thinking

        output = [r["message"]["content"] for r in res["choices"]]
        output = output if isinstance(argument.prop.prepared_input, list) else output[0]

        return output, metadata

    @staticmethod
    def _process_tool_calls(res, metadata):
        choices = res.get("choices") if isinstance(res, dict) else None
        if not choices:
            return metadata
        hit = False
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                if hit:
                    UserMessage(
                        "Multiple function calls detected in the response "
                        "but only the first one will be processed."
                    )
                    return metadata
                arguments = function.get("arguments")
                try:
                    args_dict = (
                        json.loads(arguments) if isinstance(arguments, str) else arguments or {}
                    )
                except json.JSONDecodeError:
                    args_dict = {}
                metadata["function_call"] = {
                    "name": function.get("name"),
                    "arguments": args_dict or {},
                }
                hit = True
                break
            if hit:
                break
        return metadata

    def _prepare_raw_input(self, argument):
        if not argument.prop.processed_input:
            UserMessage(
                "Need to provide a prompt instruction to the engine if raw_input is enabled.",
                raise_with=ValueError,
            )
        value = argument.prop.processed_input
        if not isinstance(value, list):
            if not isinstance(value, dict):
                value = {"role": "user", "content": str(value)}
            value = [value]
        return value

    def prepare(self, argument):
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        _non_verbose_output = (
            "<META_INSTRUCTION/>\n You will NOT output verbose preambles or post explanation, "
            'such as "Sure, let me...", "Hope that was helpful...", '
            '"Yes, I can help you with that...", etc. You will consider well formatted output, '
            "e.g. for sentences you will use punctuation, spaces, etc. or for code indentation, etc.\n"
        )

        # NOTE: single user-role concatenation matches engine_llama_cpp.py's approach —
        # many open-weights chat templates are strict about system/assistant/user
        # alternation, so we stay on the safe side and rewrite context as user content.
        user = ""

        if argument.prop.suppress_verbose_output:
            user += _non_verbose_output
        user = f"{user}\n" if user and len(user) > 0 else ""

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            user += f"<STATIC_CONTEXT/>\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            user += f"<DYNAMIC_CONTEXT/>\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            user += f"<ADDITIONAL_CONTEXT/>\n{payload!s}\n\n"

        examples = argument.prop.examples
        if examples and len(examples) > 0:
            user += f"<EXAMPLES/>\n{examples!s}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            user += f"<INSTRUCTION/>\n{argument.prop.prompt!s}\n\n"

        if argument.prop.template_suffix:
            user += f" You will only generate content for the placeholder `{argument.prop.template_suffix!s}` following the instructions and the provided context information.\n\n"

        user += str(argument.prop.processed_input)

        argument.prop.prepared_input = [
            {"role": "user", "content": user},
        ]
