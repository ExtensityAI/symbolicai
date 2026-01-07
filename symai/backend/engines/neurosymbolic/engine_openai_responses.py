import json
import logging
import re
from copy import deepcopy

import openai
import tiktoken

from ....components import SelfPrompt
from ....utils import UserMessage, encode_media_frames
from ...base import Engine
from ...mixin.openai import SUPPORTED_REASONING_MODELS, OpenAIMixin
from ...settings import SYMAI_CONFIG

logging.getLogger("openai").setLevel(logging.ERROR)
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


class ResponsesTokenizer:
    def __init__(self, model: str):
        self._model = model
        try:
            self._tiktoken = tiktoken.encoding_for_model(model)
        except Exception:
            self._tiktoken = tiktoken.get_encoding("o200k_base")

    def encode(self, text: str) -> list[int]:
        return self._tiktoken.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._tiktoken.decode(tokens)


class OpenAIResponsesEngine(Engine, OpenAIMixin):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        if api_key is not None and model is not None:
            self.config["NEUROSYMBOLIC_ENGINE_API_KEY"] = api_key
            self.config["NEUROSYMBOLIC_ENGINE_MODEL"] = model
        if self.id() != "neurosymbolic":
            return
        openai.api_key = self.config["NEUROSYMBOLIC_ENGINE_API_KEY"]
        self._prefixed_model = self.config["NEUROSYMBOLIC_ENGINE_MODEL"]
        self.model = self._strip_prefix(self._prefixed_model)
        self.seed = None
        self.name = self.__class__.__name__
        self.tokenizer = ResponsesTokenizer(model=self.model)
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()

        try:
            self.client = openai.Client(api_key=openai.api_key)
        except Exception as e:
            UserMessage(
                f"Failed to initialize OpenAI client. Caused by: {e}",
                raise_with=ValueError,
            )

    def _strip_prefix(self, model_name: str) -> str:
        return model_name.replace("responses:", "")

    def id(self) -> str:
        model = self.config.get("NEUROSYMBOLIC_ENGINE_MODEL")
        if model and model.startswith("responses:"):
            return "neurosymbolic"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "NEUROSYMBOLIC_ENGINE_API_KEY" in kwargs:
            openai.api_key = kwargs["NEUROSYMBOLIC_ENGINE_API_KEY"]
        if "NEUROSYMBOLIC_ENGINE_MODEL" in kwargs:
            self._prefixed_model = kwargs["NEUROSYMBOLIC_ENGINE_MODEL"]
            self.model = self._strip_prefix(self._prefixed_model)
        if "seed" in kwargs:
            self.seed = kwargs["seed"]

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
        if self._is_reasoning_model():
            num_tokens += 6
        else:
            num_tokens += 3
        return num_tokens

    def compute_remaining_tokens(self, prompts: list) -> int:
        val = self.compute_required_tokens(prompts)
        return min(self.max_context_tokens - val, self.max_response_tokens)

    def _is_reasoning_model(self) -> bool:
        return self.model in SUPPORTED_REASONING_MODELS or self.model in {
            "gpt-5.2-chat-latest",
            "gpt-5.1-chat-latest",
            "gpt-5-pro",
            "gpt-5.2-pro",
            "o3-pro",
        }

    def _handle_image_content(self, content: str) -> list[str]:
        def _extract_pattern(text):
            # This regular expression matches <<vision:...:>> patterns to extract embedded image references.
            pattern = r"<<vision:(.*?):>>"
            return re.findall(pattern, text)

        image_files: list[str] = []
        if "<<vision:" not in content:
            return image_files

        parts = _extract_pattern(content)
        for p in parts:
            img_ = p.strip()
            if img_.startswith("http") or img_.startswith("data:image"):
                image_files.append(img_)
            else:
                max_frames_spacing = 50
                max_used_frames = 10
                if img_.startswith("frames:"):
                    img_ = img_.replace("frames:", "")
                    max_used_frames, img_ = img_.split(":")
                    max_used_frames = int(max_used_frames)
                    if max_used_frames < 1 or max_used_frames > max_frames_spacing:
                        UserMessage(
                            f"Invalid max_used_frames value: {max_used_frames}. Expected 1-{max_frames_spacing}",
                            raise_with=ValueError,
                        )
                buffer, ext = encode_media_frames(img_)
                if len(buffer) > 1:
                    step = len(buffer) // max_frames_spacing
                    indices = list(range(0, len(buffer), step))[:max_used_frames]
                    for i in indices:
                        image_files.append(f"data:image/{ext};base64,{buffer[i]}")
                elif len(buffer) == 1:
                    image_files.append(f"data:image/{ext};base64,{buffer[0]}")
                else:
                    UserMessage("No frames found or error in encoding frames")
        return image_files

    def _remove_vision_pattern(self, text: str) -> str:
        # This regular expression matches <<vision:...:>> patterns to strip them from output text.
        pattern = r"<<vision:(.*?):>>"
        return re.sub(pattern, "", text)

    def _build_system_content(self, argument, image_files: list[str]) -> str:
        sections: list[str] = []
        sections.extend(self._verbose_section(argument))
        sections.extend(self._response_format_section(argument))
        sections.extend(self._context_sections(argument))
        sections.extend(self._payload_section(argument))
        sections.extend(self._examples_section(argument))
        sections.extend(self._instruction_section(argument, image_files))
        sections.extend(self._template_suffix_section(argument))
        return "".join(sections)

    def _verbose_section(self, argument) -> list[str]:
        if argument.prop.suppress_verbose_output:
            return [_NON_VERBOSE_OUTPUT]
        return []

    def _response_format_section(self, argument) -> list[str]:
        if (
            argument.prop.response_format
            and argument.prop.response_format.get("type") == "json_object"
        ):
            return ["<RESPONSE_FORMAT/>\nYou are a helpful assistant designed to output JSON.\n\n"]
        return []

    def _context_sections(self, argument) -> list[str]:
        sections: list[str] = []
        static_ctxt, dyn_ctxt = argument.prop.instance.global_context
        if len(static_ctxt) > 0:
            sections.append(f"<STATIC CONTEXT/>\n{static_ctxt}\n\n")
        if len(dyn_ctxt) > 0:
            sections.append(f"<DYNAMIC CONTEXT/>\n{dyn_ctxt}\n\n")
        return sections

    def _payload_section(self, argument) -> list[str]:
        if argument.prop.payload:
            return [f"<ADDITIONAL CONTEXT/>\n{argument.prop.payload!s}\n\n"]
        return []

    def _examples_section(self, argument) -> list[str]:
        examples = argument.prop.examples
        if examples and len(examples) > 0:
            return [f"<EXAMPLES/>\n{examples!s}\n\n"]
        return []

    def _instruction_section(self, argument, image_files: list[str]) -> list[str]:
        if argument.prop.prompt is None or len(argument.prop.prompt) == 0:
            return []
        val = str(argument.prop.prompt)
        if len(image_files) > 0:
            val = self._remove_vision_pattern(val)
        return [f"<INSTRUCTION/>\n{val}\n\n"]

    def _template_suffix_section(self, argument) -> list[str]:
        if argument.prop.template_suffix:
            return [
                f" You will only generate content for the placeholder `{argument.prop.template_suffix!s}` "
                "following the instructions and the provided context information.\n\n"
            ]
        return []

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

    def _apply_self_prompt_if_needed(
        self, argument, system: str, user_msg: dict, user_text: str, image_files: list[str]
    ) -> tuple[str, dict]:
        if not (
            argument.prop.instance._kwargs.get("self_prompt", False) or argument.prop.self_prompt
        ):
            return system, user_msg
        self_prompter = SelfPrompt()
        key = "developer" if self._is_reasoning_model() else "system"
        res = self_prompter({"user": user_text, key: system})
        if res is None:
            UserMessage("Self-prompting failed!", raise_with=ValueError)
        new_user_msg = self._create_user_message(res["user"], image_files)
        return res[key], new_user_msg

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

        image_files = self._handle_image_content(str(argument.prop.processed_input))
        system_content = self._build_system_content(argument, image_files)
        user_text = self._build_user_text(argument, image_files)
        user_msg = self._create_user_message(user_text, image_files)
        system_content, user_msg = self._apply_self_prompt_if_needed(
            argument, system_content, user_msg, user_text, image_files
        )

        role = "developer" if self._is_reasoning_model() else "system"
        argument.prop.prepared_input = [
            {"role": role, "content": system_content},
            user_msg,
        ]

    def _prepare_request_payload(self, messages, argument) -> dict:
        kwargs = argument.kwargs
        max_tokens = kwargs.get("max_tokens")
        max_output_tokens = kwargs.get("max_output_tokens")
        remaining_tokens = self.compute_remaining_tokens(messages)

        if max_tokens is not None:
            UserMessage(
                "'max_tokens' is deprecated in favor of 'max_output_tokens' for Responses API."
            )
            if max_tokens > self.max_response_tokens:
                max_output_tokens = remaining_tokens
            else:
                max_output_tokens = max_tokens

        if max_output_tokens is not None and max_output_tokens > self.max_response_tokens:
            UserMessage(
                f"Provided 'max_output_tokens' ({max_output_tokens}) exceeds max ({self.max_response_tokens}). "
                f"Truncating to {remaining_tokens}."
            )
            max_output_tokens = remaining_tokens

        payload: dict = {
            "model": kwargs.get("model", self.model),
            "input": messages,
        }

        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens

        if kwargs.get("temperature") is not None and not self._is_reasoning_model():
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None and not self._is_reasoning_model():
            payload["top_p"] = kwargs["top_p"]

        if self._is_reasoning_model():
            if self.model in {"gpt-5-pro", "gpt-5.2-pro"}:
                reasoning = {"effort": "high"}
            else:
                reasoning = kwargs.get("reasoning", {"effort": "medium"})
            payload["reasoning"] = reasoning

        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = self._convert_tools(tools)
            tool_choice = kwargs.get("tool_choice", "auto")
            payload["tool_choice"] = tool_choice

        if kwargs.get("response_format"):
            payload["text"] = {"format": kwargs["response_format"]}

        return payload

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

    def _extract_output_text(self, response) -> list[str]:
        outputs: list[str] = []
        for output in response.output or []:
            if output.type == "message" and output.content:
                for content in output.content:
                    if hasattr(content, "text"):
                        outputs.append(content.text)
        if not outputs and hasattr(response, "output_text") and response.output_text:
            outputs.append(response.output_text)
        return outputs

    def _process_function_calls(self, response, metadata: dict) -> dict:
        for output in response.output or []:
            if output.type == "function_call":
                try:
                    args_dict = json.loads(output.arguments)
                except json.JSONDecodeError:
                    args_dict = {}
                metadata["function_call"] = {
                    "name": output.name,
                    "arguments": args_dict,
                    "call_id": output.call_id,
                }
                break
        return metadata

    def _extract_thinking(self, response) -> str | None:
        if not self._is_reasoning_model():
            return None
        for output in response.output or []:
            if output.type == "reasoning" and hasattr(output, "summary") and output.summary:
                texts = [s.text for s in output.summary if hasattr(s, "text") and s.text]
                if texts:
                    return "\n".join(texts)
        return None

    def forward(self, argument):
        kwargs = argument.kwargs
        messages = argument.prop.prepared_input
        payload = self._prepare_request_payload(messages, argument)
        except_remedy = kwargs.get("except_remedy")

        try:
            res = self.client.responses.create(**payload)
        except Exception as e:
            if openai.api_key is None or openai.api_key == "":
                msg = "OpenAI API key is not set."
                UserMessage(msg)
                if (
                    self.config["NEUROSYMBOLIC_ENGINE_API_KEY"] is None
                    or self.config["NEUROSYMBOLIC_ENGINE_API_KEY"] == ""
                ):
                    UserMessage(msg, raise_with=ValueError)
                openai.api_key = self.config["NEUROSYMBOLIC_ENGINE_API_KEY"]

            callback = self.client.responses.create
            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                UserMessage(f"Error during generation. Caused by: {e}", raise_with=ValueError)

        metadata = {"raw_output": res}
        if payload.get("tools"):
            metadata = self._process_function_calls(res, metadata)

        thinking = self._extract_thinking(res)
        if thinking:
            metadata["thinking"] = thinking

        output = self._extract_output_text(res)
        if not output and "function_call" in metadata:
            output = [""]
        return output, metadata
