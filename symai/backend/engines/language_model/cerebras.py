import re

import tiktoken
from pydantic import TypeAdapter

from symai.backend.base import Engine
from symai.backend.usage import EngineUsageRecord
from symai.clients.cerebras.chat import (
    MODEL_SPECS as CHAT_MODEL_SPECS,
)
from symai.clients.cerebras.chat import (
    ChatCompletion,
    ChatModel,
    CreateChatCompletionRequest,
    Message,
    ReasoningEffort,
    ReasoningFormat,
    ResponseFormat,
    ServiceTier,
)
from symai.clients.cerebras.client import Client as CerebrasClient
from symai.clients.cerebras.transport import APIResponse

_NON_VERBOSE_OUTPUT = (
    "<META_INSTRUCTION/>\n"
    "You do not output anything else, like verbose preambles or post explanation, such as "
    '"Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. '
    "Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use "
    "indentation, etc. Never add meta instructions information to your output!\n\n"
)


class LanguageModelEngine(Engine):
    provider = "cerebras"
    capability = "language_model"

    def __init__(self, *, client: CerebrasClient, model: ChatModel):
        super().__init__()
        try:
            self.model_spec = CHAT_MODEL_SPECS[model]
        except KeyError as e:
            msg = f"Unsupported model: {model}"
            raise ValueError(msg) from e

        self.client = client
        self.model = model
        self.seed = None
        self.name = self.__class__.__name__
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.max_context_tokens = self.model_spec.context_tokens
        self.max_response_tokens = self.model_spec.response_tokens

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
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
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def build_request(self, argument) -> CreateChatCompletionRequest:
        unsupported = {"stream", "stream_options", "tools", "tool_choice"} & set(argument.kwargs)
        if unsupported:
            msg = (
                "Cerebras integration does not support these request options: "
                f"{sorted(unsupported)}"
            )
            raise ValueError(msg)

        request_kwargs = set(CreateChatCompletionRequest.model_fields) - {"messages", "model"}
        payload = self.collect_request_kwargs(argument, request_kwargs | {"max_tokens"})
        if "max_completion_tokens" not in payload and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        model = self.model
        reasoning = self.model_spec.reasoning
        supported_efforts = reasoning.efforts if reasoning is not None else ()
        reasoning_effort = payload.get("reasoning_effort")
        if reasoning_effort is not None and reasoning_effort not in supported_efforts:
            msg = (
                f"Unsupported reasoning_effort for Cerebras model {model}: "
                f"{reasoning_effort}. Supported values: "
                f"{[effort.value for effort in supported_efforts]}"
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
        return CreateChatCompletionRequest.model_validate(payload)

    def call_request(self, request: CreateChatCompletionRequest) -> APIResponse[ChatCompletion]:
        return self.client.create_chat_completion(request)

    def usage_record_from_metadata(self, metadata: dict) -> EngineUsageRecord | None:
        usage = metadata["raw_output"].usage
        if usage is None:
            return None

        return EngineUsageRecord(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            total_tokens=usage.total_tokens or 0,
        )

    def parse_response(self, response: APIResponse[ChatCompletion]):
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
            result = self.self_prompt({"user": user_prompt["content"], "system": system_message})
            if result is None:
                msg = "Self-prompting failed!"
                raise ValueError(msg)
            return result["system"], {"role": "user", "content": result["user"]}
        return system_message, user_prompt
