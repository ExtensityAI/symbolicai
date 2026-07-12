import re
from copy import deepcopy

import httpx
import tiktoken
from pydantic import TypeAdapter

from symai.backend.base import Engine
from symai.backend.integrations.cerebras.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    ReasoningEffort,
    ReasoningFormat,
    ResponseFormat,
    ServiceTier,
)
from symai.backend.integrations.cerebras.client import Client as CerebrasClient
from symai.backend.integrations.cerebras.response import Response
from symai.backend.mixin.cerebras import SUPPORTED_CEREBRAS_MODELS, CerebrasMixin
from symai.backend.settings import SYMAI_CONFIG
from symai.components import SelfPrompt

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
        http_client: httpx.Client | None = None,
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

        self.http_client = http_client

    def id(self) -> str:
        model_name = self.config.get("NEUROSYMBOLIC_ENGINE_MODEL")
        if model_name and model_name in SUPPORTED_CEREBRAS_MODELS:
            return "neurosymbolic"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "NEUROSYMBOLIC_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["NEUROSYMBOLIC_ENGINE_API_KEY"]
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

    def forward(self, argument):  # pyright: ignore[reportIncompatibleMethodOverride]
        if not self.api_key:
            msg = (
                "Cerebras API key is not set. Please set it in the config file or "
                "pass it when constructing the engine."
            )
            raise ValueError(msg)

        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def build_request(self, argument) -> ChatRequest:
        unsupported = {"stream", "stream_options", "tools", "tool_choice"} & set(argument.kwargs)
        if unsupported:
            msg = (
                "Cerebras integration does not support these request options: "
                f"{sorted(unsupported)}"
            )
            raise ValueError(msg)

        request_kwargs = set(ChatRequest.model_fields) - {"messages"}
        payload = self.collect_request_kwargs(argument, request_kwargs | {"max_tokens"})
        if "max_completion_tokens" not in payload and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        model = self.cerebras_strip_prefix(payload.get("model", self.model))
        model_spec = self.cerebras_model_spec_for(model)
        reasoning_effort = payload.get("reasoning_effort")
        if reasoning_effort is not None and reasoning_effort not in model_spec.reasoning_efforts:
            msg = (
                f"Unsupported reasoning_effort for Cerebras model {model}: "
                f"{reasoning_effort}. Supported values: {list(model_spec.reasoning_efforts)}"
            )
            raise ValueError(msg)

        payload["model"] = model
        payload["messages"] = tuple(
            TypeAdapter(Message).validate_python(message, strict=False)
            for message in argument.prop.prepared_input
        )
        if "seed" not in payload and self.seed is not None:
            payload["seed"] = self.seed
        if "stop" in payload and not payload["stop"]:
            payload["stop"] = None
        if isinstance(reasoning_effort, str):
            payload["reasoning_effort"] = ReasoningEffort(reasoning_effort)
        if isinstance(payload.get("reasoning_format"), str):
            payload["reasoning_format"] = ReasoningFormat(payload["reasoning_format"])
        if isinstance(payload.get("service_tier"), str):
            payload["service_tier"] = ServiceTier(payload["service_tier"])

        response_format = self._normalize_response_format(payload.get("response_format"))
        if isinstance(response_format, dict):
            response_format = TypeAdapter(ResponseFormat).validate_python(
                response_format,
                strict=False,
            )
        payload["response_format"] = response_format
        return ChatRequest.model_validate(payload)

    def call_request(self, request: ChatRequest) -> Response[ChatResponse]:
        if self.http_client is not None:
            return CerebrasClient(
                api_key=self.api_key,
                http_client=self.http_client,
            ).chat(request)

        with httpx.Client(timeout=self.client_timeout) as http_client:
            return CerebrasClient(
                api_key=self.api_key,
                http_client=http_client,
            ).chat(request)

    def parse_response(self, response: Response[ChatResponse]):
        raw_output = response.data
        if not raw_output.choices:
            msg = "Cerebras response did not contain any choices"
            raise ValueError(msg)

        outputs: list[str] = []
        thinking_content: str | None = None
        for choice in raw_output.choices:
            message = choice.message
            if message is None:
                outputs.append("")
                continue
            outputs.append(message.content or "")
            if thinking_content is None and message.reasoning:
                thinking_content = message.reasoning

        if thinking_content is None:
            thinking_content, outputs = self._extract_thinking_content(outputs)
        else:
            _, outputs = self._extract_thinking_content(outputs)

        metadata: dict = {
            "raw_output": raw_output,
            "response": response,
        }
        if thinking_content:
            metadata["thinking"] = thinking_content
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
