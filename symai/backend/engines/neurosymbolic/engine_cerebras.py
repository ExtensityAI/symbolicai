import json
import logging
import re
from copy import deepcopy

from cerebras.cloud.sdk import Cerebras

from ....components import SelfPrompt
from ....core_ext import retry
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

logging.getLogger("cerebras").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


_NON_VERBOSE_OUTPUT = (
    "<META_INSTRUCTION/>\n"
    "You do not output anything else, like verbose preambles or post explanation, such as "
    '"Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. '
    "Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use "
    "indentation, etc. Never add meta instructions information to your output!\n\n"
)


class CerebrasEngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
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

        try:
            self.client = Cerebras(api_key=self.api_key)
        except Exception as exc:
            UserMessage(
                f"Failed to initialize Cerebras client. Please check your Cerebras SDK installation. Caused by: {exc}",
                raise_with=ValueError,
            )

    def id(self) -> str:
        model_name = self.config.get("NEUROSYMBOLIC_ENGINE_MODEL")
        if model_name and model_name.startswith("cerebras"):
            return "neurosymbolic"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "NEUROSYMBOLIC_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["NEUROSYMBOLIC_ENGINE_API_KEY"]
            try:
                self.client = Cerebras(api_key=self.api_key)
            except Exception as exc:
                UserMessage(
                    f"Failed to reinitialize Cerebras client. Caused by: {exc}",
                    raise_with=ValueError,
                )
        if "NEUROSYMBOLIC_ENGINE_MODEL" in kwargs:
            self.model = kwargs["NEUROSYMBOLIC_ENGINE_MODEL"]
        if "seed" in kwargs:
            self.seed = kwargs["seed"]

    def compute_required_tokens(self, _messages):
        UserMessage(
            "Token counting not implemented for this engine.", raise_with=NotImplementedError
        )

    def compute_remaining_tokens(self, _prompts: list) -> int:
        UserMessage(
            "Token counting not implemented for this engine.", raise_with=NotImplementedError
        )

    def _handle_prefix(self, model_name: str) -> str:
        """Handle prefix for model name."""
        return model_name.replace("cerebras:", "")

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

    # cumulative wait time is < 30s
    @retry(tries=8, delay=0.5, backoff=1.5, max_delay=5, jitter=(0, 0.5))
    def forward(self, argument):
        kwargs = argument.kwargs
        messages = argument.prop.prepared_input
        payload = self._prepare_request_payload(messages, argument)
        except_remedy = kwargs.get("except_remedy")

        try:
            res = self.client.chat.completions.create(**payload)
        except Exception as exc:  # pragma: no cover - defensive path
            res = self._handle_forward_exception(exc, argument, kwargs, except_remedy)

        return self._build_outputs_and_metadata(res, payload)

    def _handle_forward_exception(self, exc, argument, kwargs, except_remedy):
        if self.api_key is None or self.api_key == "":
            msg = (
                "Cerebras API key is not set. Please set it in the config file or "
                "pass it as an argument to the command method."
            )
            UserMessage(msg)
            config_key = self.config.get("NEUROSYMBOLIC_ENGINE_API_KEY")
            if config_key is None or config_key == "":
                UserMessage(msg, raise_with=ValueError)
            self.api_key = config_key
            try:
                self.client = Cerebras(api_key=self.api_key)
            except Exception as inner_exc:
                UserMessage(
                    f"Failed to initialize Cerebras client after missing API key. Caused by: {inner_exc}",
                    raise_with=ValueError,
                )

        callback = self.client.chat.completions.create
        kwargs["model"] = (
            self._handle_prefix(kwargs["model"])
            if "model" in kwargs
            else self._handle_prefix(self.model)
        )

        if except_remedy is not None:
            return except_remedy(self, exc, callback, argument)

        UserMessage(f"Error during generation. Caused by: {exc}", raise_with=ValueError)
        return None

    def _build_outputs_and_metadata(self, res, payload):
        metadata: dict = {"raw_output": res}
        if payload.get("tools"):
            metadata = self._process_function_calls(res, metadata)

        outputs: list[str] = []
        thinking_content: str | None = None

        for choice in res.choices:
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

        return outputs, metadata

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
                UserMessage("Self-prompting failed!", raise_with=ValueError)
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
                        UserMessage(
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

    def _prepare_request_payload(self, messages, argument):
        """Prepares the request payload from the argument."""
        kwargs = argument.kwargs

        n = kwargs.get("n", 1)
        if n > 1:
            UserMessage(
                "If N is supplied, it must be equal to 1. We default to 1 to avoid unexpected batch behavior."
            )
            n = 1

        response_format = kwargs.get("response_format")

        return {
            "messages": messages,
            "model": self._handle_prefix(kwargs.get("model", self.model)),
            "max_completion_tokens": kwargs.get("max_completion_tokens"),
            "stop": kwargs.get("stop"),
            "temperature": kwargs.get("temperature", 1),
            "top_p": kwargs.get("top_p", 1),
            "n": n,
            "tools": kwargs.get("tools"),
            "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
            "response_format": response_format,
            "reasoning_effort": kwargs.get("reasoning_effort"),
            "disable_reasoning": kwargs.get("disable_reasoning"),
            "stream": kwargs.get("stream", False),
        }
