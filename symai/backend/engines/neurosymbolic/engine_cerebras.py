import json
import logging
import re
from copy import deepcopy

import tiktoken
from cerebras.cloud.sdk import Cerebras

from symai.backend.base import Engine
from symai.backend.mixin.cerebras import (
    SUPPORTED_CEREBRAS_MODELS,
    CerebrasChatCreateCallOptions,
    CerebrasChatCreatePayload,
    CerebrasChatRequest,
    CerebrasMixin,
)
from symai.backend.settings import SYMAI_CONFIG
from symai.components import SelfPrompt
from symai.utils import silence_noisy_loggers

silence_noisy_loggers("cerebras", "hpack")

logger = logging.getLogger(__name__)


_NON_VERBOSE_OUTPUT = (
    "<META_INSTRUCTION/>\n"
    "You do not output anything else, like verbose preambles or post explanation, such as "
    '"Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. '
    "Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use "
    "indentation, etc. Never add meta instructions information to your output!\n\n"
)


class CerebrasEngine(CerebrasMixin, Engine):
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
            # Do not initialize if not neurosymbolic; avoids conflict with llama.cpp check in
            # EngineRepository.register_from_package.
            return

        self.api_key = self.config["NEUROSYMBOLIC_ENGINE_API_KEY"]
        self.model = self.config["NEUROSYMBOLIC_ENGINE_MODEL"]
        self.seed = None
        self.name = self.__class__.__name__
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()

        try:
            self.client = self._build_cerebras_client(self.api_key)
        except Exception as exc:
            msg = f"Failed to initialize Cerebras client. Please check your Cerebras SDK installation. Caused by: {exc}"
            raise ValueError(msg) from exc

    def _build_cerebras_client(self, api_key: str | None):
        return Cerebras(**self._build_client_kwargs({"api_key": api_key}))

    def id(self) -> str:
        model_name = self.config.get("NEUROSYMBOLIC_ENGINE_MODEL")
        if model_name and model_name in SUPPORTED_CEREBRAS_MODELS:
            return "neurosymbolic"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "NEUROSYMBOLIC_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["NEUROSYMBOLIC_ENGINE_API_KEY"]
            try:
                self.client = self._build_cerebras_client(self.api_key)
            except Exception as exc:
                msg = f"Failed to reinitialize Cerebras client. Caused by: {exc}"
                raise ValueError(msg) from exc
        if "NEUROSYMBOLIC_ENGINE_MODEL" in kwargs:
            self.model = kwargs["NEUROSYMBOLIC_ENGINE_MODEL"]
        if "seed" in kwargs:
            self.seed = kwargs["seed"]

    def compute_required_tokens(self, messages):
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(self.tokenizer.encode(value, disallowed_special=()))
                else:
                    for v in value:
                        if v["type"] == "text":
                            num_tokens += len(
                                self.tokenizer.encode(v["text"], disallowed_special=())
                            )
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def compute_remaining_tokens(self, prompts: list) -> int:
        val = self.compute_required_tokens(prompts)
        return min(self.max_context_tokens - val, self.max_response_tokens)

    def _handle_prefix(self, model_name: str) -> str:
        """Handle prefix for model name."""
        return self.cerebras_strip_prefix(model_name)

    @staticmethod
    def _normalize_response_format(response_format: dict | None) -> dict | None:
        """Normalize response_format to the Cerebras/OpenAI expected structure.

        Cerebras expects json_schema as:
            {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": true}}
        but callers may pass the flat form:
            {"type": "json_schema", "name": "...", "schema": {...}}
        """
        if not isinstance(response_format, dict):
            return response_format
        if response_format.get("type") != "json_schema":
            return response_format
        if "json_schema" in response_format:
            return response_format  # already in correct format
        # Reshape flat form into nested form
        inner = {k: v for k, v in response_format.items() if k != "type"}
        inner.setdefault("strict", True)
        return {"type": "json_schema", "json_schema": inner}

    def _extract_thinking_content(self, outputs: list[str]) -> tuple[str | None, list[str]]:
        """Extract thinking content from textual output using <think>...</think> tags if present."""
        if not outputs:
            return None, outputs

        content = outputs[0]
        if not content:
            return None, outputs

        # This regular expression matches a <think>...</think> block and captures any content between the tags,
        # including newlines, so that we can separate internal reasoning text from the user-facing answer.
        think_pattern = r"<think>(.*?)</think>"
        match = re.search(think_pattern, content, re.DOTALL)

        thinking_content = None
        if match:
            thinking_content = match.group(1).strip() or None

        cleaned_content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()
        cleaned_outputs = [cleaned_content, *outputs[1:]]

        return thinking_content, cleaned_outputs

    def forward(self, argument):
        kwargs = argument.kwargs
        request = self.build_request(argument)
        except_remedy = kwargs.get("except_remedy")

        try:
            res = self.call_request(request)
        except Exception as exc:  # pragma: no cover - defensive path
            res = self._handle_forward_exception(exc, argument, kwargs, except_remedy, request)

        return self.parse_response(res)

    def _handle_forward_exception(self, exc, argument, kwargs, except_remedy, request):
        if self.api_key is None or self.api_key == "":
            msg = (
                "Cerebras API key is not set. Please set it in the config file or "
                "pass it as an argument to the command method."
            )
            logger.warning(msg)
            config_key = self.config.get("NEUROSYMBOLIC_ENGINE_API_KEY")
            if config_key is None or config_key == "":
                raise ValueError(msg)
            self.api_key = config_key
            try:
                self.client = self._build_cerebras_client(self.api_key)
            except Exception as inner_exc:
                msg = f"Failed to initialize Cerebras client after missing API key. Caused by: {inner_exc}"
                raise ValueError(msg) from inner_exc

        callback = self.client.chat.completions.create
        kwargs["model"] = request.body()["model"]

        if except_remedy is not None:
            return except_remedy(self, exc, callback, argument)

        msg = f"Error during generation. Caused by: {exc}"
        raise ValueError(msg)

    def build_request(self, argument) -> CerebrasChatRequest:
        request_kwargs = set(CerebrasChatCreatePayload.model_fields) | set(
            CerebrasChatCreateCallOptions.model_fields
        )
        payload_kwargs = self.collect_request_kwargs(argument, request_kwargs)
        call_options = {
            key: payload_kwargs.pop(key)
            for key in CerebrasChatCreateCallOptions.model_fields
            if key in payload_kwargs
        }

        model = self.cerebras_strip_prefix(payload_kwargs.get("model", self.model))
        model_spec = self.cerebras_model_spec_for(model)
        reasoning_effort = payload_kwargs.get("reasoning_effort")
        if reasoning_effort is not None and reasoning_effort not in model_spec.reasoning_efforts:
            msg = (
                f"Unsupported reasoning_effort for Cerebras model {model}: "
                f"{reasoning_effort}. Supported values: {list(model_spec.reasoning_efforts)}"
            )
            raise ValueError(msg)
        payload_kwargs["model"] = model
        payload_kwargs["messages"] = argument.prop.prepared_input
        if "seed" not in payload_kwargs and self.seed is not None:
            payload_kwargs["seed"] = self.seed
        if "stop" in payload_kwargs and not payload_kwargs["stop"]:
            payload_kwargs["stop"] = None
        payload_kwargs["temperature"] = payload_kwargs.get("temperature", 1)
        payload_kwargs["top_p"] = payload_kwargs.get("top_p", 1)
        payload_kwargs["stream"] = payload_kwargs.get("stream", False)
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

        payload = CerebrasChatCreatePayload.model_validate(payload_kwargs)
        options = None
        if call_options:
            options = CerebrasChatCreateCallOptions.model_validate(call_options)

        return CerebrasChatRequest(
            provider="cerebras",
            operation="chat.completions.create",
            payload=payload,
            call_options=options,
        )

    def call_request(self, request: CerebrasChatRequest):
        return self.client.chat.completions.create(**request.kwargs())

    def parse_response(self, response):
        metadata: dict = {"raw_output": response}
        metadata = self._process_function_calls(response, metadata)

        outputs: list[str] = []
        thinking_content: str | None = None

        for choice in response.choices:
            message = choice.message
            outputs.append(getattr(message, "content", "") or "")
            if thinking_content is None:
                reasoning = getattr(message, "reasoning", None)
                if reasoning:
                    thinking_content = reasoning

        if thinking_content is None:
            thinking_content, outputs = self._extract_thinking_content(outputs)
        else:
            _, outputs = self._extract_thinking_content(outputs)

        if thinking_content:
            metadata["thinking"] = thinking_content

        if not outputs and "function_call" in metadata:
            outputs = [""]

        return outputs, metadata

    def _prepare_raw_input(self, argument):
        if not argument.prop.processed_input:
            msg = "Need to provide a prompt instruction to the engine if raw_input is enabled."
            raise ValueError(msg)
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
        self._validate_response_format(argument)

        system_message = self._build_system_message(argument)
        user_content = self._build_user_content(argument)
        user_prompt = {"role": "user", "content": user_content}
        system_message, user_prompt = self._apply_self_prompt_if_needed(
            argument, system_message, user_prompt
        )

        argument.prop.prepared_input = [
            {"role": "system", "content": system_message},
            user_prompt,
        ]

    def _validate_response_format(self, argument) -> None:
        if argument.prop.response_format:
            response_format = argument.prop.response_format
            assert response_format.get("type") is not None, (
                'Expected format `{ "type": "json_object" }` for JSON mode. '
                "See Cerebras structured outputs documentation for details."
            )

    def _build_system_message(self, argument) -> str:
        system_message: str = ""
        if argument.prop.suppress_verbose_output:
            system_message += _NON_VERBOSE_OUTPUT
        if system_message:
            system_message = f"{system_message}\n"

        ref = argument.prop.instance
        static_context, dynamic_context = ref.global_context
        if len(static_context) > 0:
            system_message += f"<STATIC CONTEXT/>\n{static_context}\n\n"

        if len(dynamic_context) > 0:
            system_message += f"<DYNAMIC CONTEXT/>\n{dynamic_context}\n\n"

        if argument.prop.payload:
            system_message += f"<ADDITIONAL CONTEXT/>\n{argument.prop.payload!s}\n\n"

        examples = argument.prop.examples
        if examples and len(examples) > 0:
            system_message += f"<EXAMPLES/>\n{examples!s}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            prompt_value = str(argument.prop.prompt)
            system_message += f"<INSTRUCTION/>\n{prompt_value}\n\n"

        if argument.prop.template_suffix:
            system_message += (
                " You will only generate content for the placeholder "
                f"`{argument.prop.template_suffix!s}` following the instructions and the provided context information.\n\n"
            )

        return system_message

    def _build_user_content(self, argument) -> str:
        return str(argument.prop.processed_input)

    def _apply_self_prompt_if_needed(self, argument, system_message, user_prompt):
        if argument.prop.instance._kwargs.get("self_prompt", False) or argument.prop.self_prompt:
            self_prompter = SelfPrompt()
            result = self_prompter({"user": user_prompt["content"], "system": system_message})
            if result is None:
                msg = "Self-prompting failed!"
                raise ValueError(msg)
            return result["system"], {"role": "user", "content": result["user"]}
        return system_message, user_prompt

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
