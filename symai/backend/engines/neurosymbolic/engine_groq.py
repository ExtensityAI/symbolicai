import json
import logging
import re
from copy import deepcopy

import openai

from symai.backend.base import Engine
from symai.backend.mixin.groq import (
    GROQ_UNSUPPORTED_REQUEST_KWARGS,
    SUPPORTED_GROQ_MODELS,
    GroqChatCreateAliases,
    GroqChatCreateCallOptions,
    GroqChatCreatePayload,
    GroqChatRequest,
    GroqMixin,
)
from symai.backend.settings import SYMAI_CONFIG
from symai.components import SelfPrompt
from symai.utils import silence_noisy_loggers

silence_noisy_loggers("openai")

logger = logging.getLogger(__name__)


_NON_VERBOSE_OUTPUT = (
    "<META_INSTRUCTION/>\n"
    "You do not output anything else, like verbose preambles or post explanation, such as "
    '"Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. '
    "Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use "
    "indentation, etc. Never add meta instructions information to your output!\n\n"
)


class GroqEngine(Engine, GroqMixin):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
    ):
        super().__init__(client_timeout=client_timeout, client_max_retries=client_max_retries)
        self.config = deepcopy(SYMAI_CONFIG)
        # In case we use EngineRepository.register to inject the api_key and model => dynamically change the engine at runtime
        if api_key is not None and model is not None:
            self.config["NEUROSYMBOLIC_ENGINE_API_KEY"] = api_key
            self.config["NEUROSYMBOLIC_ENGINE_MODEL"] = model
        if self.id() != "neurosymbolic":
            return  # do not initialize if not neurosymbolic; avoids conflict with llama.cpp check in EngineRepository.register_from_package
        openai.api_key = self.config["NEUROSYMBOLIC_ENGINE_API_KEY"]
        self.model = self.config[
            "NEUROSYMBOLIC_ENGINE_MODEL"
        ]  # Keep the original config name to avoid confusion in downstream tasks
        self.seed = None
        self.name = self.__class__.__name__
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()

        try:
            self.client = openai.OpenAI(
                **self._build_client_kwargs(
                    {
                        "api_key": openai.api_key,
                        "base_url": "https://api.groq.com/openai/v1",
                    }
                )
            )
        except Exception as e:
            msg = f"Failed to initialize OpenAI client. Please check your OpenAI library version. Caused by: {e}"
            raise ValueError(msg) from e

    def id(self) -> str:
        if self.config.get("NEUROSYMBOLIC_ENGINE_MODEL") in SUPPORTED_GROQ_MODELS:
            return "neurosymbolic"
        return super().id()  # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "NEUROSYMBOLIC_ENGINE_API_KEY" in kwargs:
            openai.api_key = kwargs["NEUROSYMBOLIC_ENGINE_API_KEY"]
        if "NEUROSYMBOLIC_ENGINE_MODEL" in kwargs:
            self.model = kwargs["NEUROSYMBOLIC_ENGINE_MODEL"]
        if "seed" in kwargs:
            self.seed = kwargs["seed"]

    def compute_required_tokens(self, _messages):
        msg = "Token counting not implemented for this engine."
        raise NotImplementedError(msg)

    def compute_remaining_tokens(self, _prompts: list) -> int:
        msg = "Token counting not implemented for this engine."
        raise NotImplementedError(msg)

    def _handle_prefix(self, model_name: str) -> str:
        """Handle prefix for model name."""
        return self.groq_strip_prefix(model_name)

    def _extract_thinking_content(self, output: list[str]) -> tuple[str | None, list[str]]:
        """Extract thinking content from model output if present and return cleaned output."""
        if not output or len(output) == 0:
            return None, output

        content = output[0]
        if not content:
            return None, output

        # NOTE: Matches Groq reasoning text wrapped in <think>...</think> tags.
        think_pattern = r"<think>(.*?)</think>"
        match = re.search(think_pattern, content, re.DOTALL)

        thinking_content = None
        if match:
            thinking_content = match.group(1).strip()
            if not thinking_content:
                thinking_content = None

        cleaned_content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()
        cleaned_output = [cleaned_content, *output[1:]]

        return thinking_content, cleaned_output

    def forward(self, argument):
        kwargs = argument.kwargs
        request = self.build_request(argument)
        except_remedy = kwargs.get("except_remedy")

        try:
            res = self.call_request(request)

        except Exception as e:
            if openai.api_key is None or openai.api_key == "":
                msg = "Groq API key is not set. Please set it in the config file or pass it as an argument to the command method."
                logger.warning(msg)
                if (
                    self.config["NEUROSYMBOLIC_ENGINE_API_KEY"] is None
                    or self.config["NEUROSYMBOLIC_ENGINE_API_KEY"] == ""
                ):
                    raise ValueError(msg) from e
                openai.api_key = self.config["NEUROSYMBOLIC_ENGINE_API_KEY"]

            callback = self.client.chat.completions.create
            kwargs["model"] = request.body()["model"]

            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                msg = f"Error during generation. Caused by: {e}"
                raise ValueError(msg) from e

        return self.parse_response(res)

    def build_request(self, argument) -> GroqChatRequest:
        request_kwargs = (
            set(GroqChatCreatePayload.model_fields)
            | set(GroqChatCreateCallOptions.model_fields)
            | set(GroqChatCreateAliases.model_fields)
            | set(GROQ_UNSUPPORTED_REQUEST_KWARGS)
        )
        payload_kwargs = self.collect_request_kwargs(argument, request_kwargs)
        call_options = {
            key: payload_kwargs.pop(key)
            for key in GroqChatCreateCallOptions.model_fields
            if key in payload_kwargs
        }
        alias_kwargs = {
            key: payload_kwargs.pop(key)
            for key in GroqChatCreateAliases.model_fields
            if key in payload_kwargs
        }
        for key in sorted(GROQ_UNSUPPORTED_REQUEST_KWARGS):
            if key in payload_kwargs:
                warning_message = (
                    f"The parameter {key} is not supported by the Groq API. It will be ignored."
                )
                logger.warning(warning_message)
                payload_kwargs.pop(key)

        aliases = GroqChatCreateAliases.model_validate(alias_kwargs)
        model = self.groq_strip_prefix(payload_kwargs.get("model", self.model))
        spec = self.groq_model_spec_for(model)

        payload_kwargs["model"] = model
        payload_kwargs["messages"] = argument.prop.prepared_input
        if aliases.max_tokens is not None and "max_completion_tokens" not in payload_kwargs:
            payload_kwargs["max_completion_tokens"] = aliases.max_tokens
        if "seed" not in payload_kwargs and self.seed is not None:
            payload_kwargs["seed"] = self.seed
        if "stop" in payload_kwargs and not payload_kwargs["stop"]:
            payload_kwargs["stop"] = None
        payload_kwargs["temperature"] = payload_kwargs.get("temperature", 1)
        payload_kwargs["frequency_penalty"] = payload_kwargs.get("frequency_penalty", 0)
        payload_kwargs["presence_penalty"] = payload_kwargs.get("presence_penalty", 0)
        payload_kwargs["service_tier"] = payload_kwargs.get("service_tier", "on_demand")
        payload_kwargs["top_p"] = payload_kwargs.get("top_p", 1)

        n = payload_kwargs.get("n", 1)
        if n > 1:
            logger.warning(
                "If N is supplied, it must be equal to 1. We default to 1 to not crash your program."
            )
            n = 1
        payload_kwargs["n"] = n

        if "reasoning_effort" not in payload_kwargs:
            payload_kwargs["reasoning_effort"] = spec.default_reasoning_effort
        if not self.groq_supports_reasoning_effort(payload_kwargs["reasoning_effort"]):
            msg = (
                f"Unsupported reasoning_effort for Groq model {model}: "
                f"{payload_kwargs['reasoning_effort']}"
            )
            raise ValueError(msg)

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

        payload = GroqChatCreatePayload.model_validate(payload_kwargs)
        options = None
        if call_options:
            options = GroqChatCreateCallOptions.model_validate(call_options)

        return GroqChatRequest(
            provider="groq",
            operation="chat.completions.create",
            payload=payload,
            call_options=options,
        )

    def call_request(self, request: GroqChatRequest):
        return self.client.chat.completions.create(**request.kwargs())

    def parse_response(self, response):
        metadata = {"raw_output": response}
        metadata = self._process_function_calls(response, metadata)

        output = [choice.message.content or "" for choice in response.choices]
        thinking, output = self._extract_thinking_content(output)
        if thinking:
            metadata["thinking"] = thinking
        if not output and "function_call" in metadata:
            output = [""]

        return output, metadata

    def _prepare_raw_input(self, argument):
        if not argument.prop.processed_input:
            msg = "Need to provide a prompt instruction to the engine if raw_input is enabled."
            raise ValueError(msg)
        value = argument.prop.processed_input
        # convert to dict if not already
        if not isinstance(value, list):
            if not isinstance(value, dict):
                value = {"role": "user", "content": str(value)}
            value = [value]
        return value

    def prepare(self, argument):
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return
        self._validate_response_format(argument)

        system = self._build_system_message(argument)
        user_content = self._build_user_content(argument)
        user_prompt = {"role": "user", "content": user_content}
        system, user_prompt = self._apply_self_prompt_if_needed(argument, system, user_prompt)

        argument.prop.prepared_input = [
            {"role": "system", "content": system},
            user_prompt,
        ]

    def _validate_response_format(self, argument) -> None:
        if argument.prop.response_format:
            response_format = argument.prop.response_format
            assert response_format.get("type") is not None, (
                'Expected format `{ "type": "json_object" }`! We are using the OpenAI compatible API for Groq. '
                "See more here: https://console.groq.com/docs/tool-use"
            )

    def _build_system_message(self, argument) -> str:
        system = ""
        if argument.prop.suppress_verbose_output:
            system += _NON_VERBOSE_OUTPUT
        if system:
            system = f"{system}\n"

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"<STATIC CONTEXT/>\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"<DYNAMIC CONTEXT/>\n{dyn_ctxt}\n\n"

        if argument.prop.payload:
            system += f"<ADDITIONAL CONTEXT/>\n{argument.prop.payload!s}\n\n"

        examples = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"<EXAMPLES/>\n{examples!s}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            system += f"<INSTRUCTION/>\n{val}\n\n"

        if argument.prop.template_suffix:
            system += (
                " You will only generate content for the placeholder "
                f"`{argument.prop.template_suffix!s}` following the instructions and the provided context information.\n\n"
            )

        return system

    def _build_user_content(self, argument) -> str:
        return str(argument.prop.processed_input)

    def _apply_self_prompt_if_needed(self, argument, system, user_prompt):
        if argument.prop.instance._kwargs.get("self_prompt", False) or argument.prop.self_prompt:
            self_prompter = SelfPrompt()
            res = self_prompter({"user": user_prompt["content"], "system": system})
            if res is None:
                msg = "Self-prompting failed!"
                raise ValueError(msg)
            return res["system"], {"role": "user", "content": res["user"]}
        return system, user_prompt

    def _process_function_calls(self, res, metadata):
        hit = False
        if (
            hasattr(res, "choices")
            and res.choices
            and hasattr(res.choices[0], "message")
            and res.choices[0].message
            and hasattr(res.choices[0].message, "tool_calls")
            and res.choices[0].message.tool_calls
        ):
            for tool_call in res.choices[0].message.tool_calls:
                if hasattr(tool_call, "function") and tool_call.function:
                    if hit:
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
                    hit = True
        return metadata
