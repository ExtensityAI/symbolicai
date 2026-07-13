import logging
import re
from copy import deepcopy

import httpx
import tiktoken
from pydantic import TypeAdapter

from symai.backend.base import Engine
from symai.backend.integrations.openai.client import Client as OpenAIClient
from symai.backend.integrations.openai.responses import (
    ContextCompaction,
    Conversation,
    CreateResponseRequest,
    InputMessage,
    ModerationConfig,
    OutputMessage,
    OutputText,
    PromptCacheOptions,
    PromptReference,
    ReasoningConfig,
    Response,
    ResponseStatus,
    ServiceTier,
    TextConfig,
    Truncation,
)
from symai.backend.integrations.openai.transport import APIResponse
from symai.backend.mixin.openai import SUPPORTED_OPENAI_MODELS, OpenAIMixin
from symai.backend.settings import SYMAI_CONFIG
from symai.components import SelfPrompt
from symai.utils import encode_media_frames

logger = logging.getLogger(__name__)

_NON_VERBOSE_OUTPUT = (
    "<META_INSTRUCTION/>\n"
    "You do not output anything else, like verbose preambles or post explanation, such as "
    '"Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. '
    "Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use "
    "indentation, etc. Never add meta instructions information to your output!\n\n"
)


class ResponsesTokenizer:
    def __init__(self, model: str, tokenizer_name: str):
        self._model = model
        try:
            self._tiktoken = tiktoken.encoding_for_model(model)
        except Exception:
            self._tiktoken = tiktoken.get_encoding(tokenizer_name)

    def encode(self, text: str) -> list[int]:
        return self._tiktoken.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._tiktoken.decode(tokens)


class OpenAIResponsesEngine(Engine, OpenAIMixin):
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
        if api_key is not None:
            self.config["NEUROSYMBOLIC_ENGINE_API_KEY"] = api_key
        if model is not None:
            self.config["NEUROSYMBOLIC_ENGINE_MODEL"] = model
        if self.id() != "neurosymbolic":
            return
        self.api_key = self.config["NEUROSYMBOLIC_ENGINE_API_KEY"]
        self.model = self._strip_prefix(self.config["NEUROSYMBOLIC_ENGINE_MODEL"])
        self.seed = None
        self.name = self.__class__.__name__
        self.tokenizer = ResponsesTokenizer(
            model=self.model,
            tokenizer_name=self.openai_tokenizer_name(),
        )
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.http_client = http_client

    def _strip_prefix(self, model_name: str) -> str:
        if model_name.startswith("openai:"):
            return model_name.removeprefix("openai:")
        return model_name

    def id(self) -> str:
        model = self.config.get("NEUROSYMBOLIC_ENGINE_MODEL")
        if model and model in SUPPORTED_OPENAI_MODELS:
            return "neurosymbolic"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "NEUROSYMBOLIC_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["NEUROSYMBOLIC_ENGINE_API_KEY"]
        if "NEUROSYMBOLIC_ENGINE_MODEL" in kwargs:
            self.model = self._strip_prefix(kwargs["NEUROSYMBOLIC_ENGINE_MODEL"])
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
        if self.is_openai_reasoning_model():
            num_tokens += 6
        else:
            num_tokens += 3
        return num_tokens

    def compute_remaining_tokens(self, prompts: list) -> int:
        val = self.compute_required_tokens(prompts)
        return min(self.max_context_tokens - val, self.max_response_tokens)

    def _handle_image_content(self, content: str) -> list[str]:
        def _extract_pattern(text):
            # This regular expression matches <<vision:...:>> patterns to extract embedded image references.
            pattern = r"<<vision:(.*?):>>"
            return re.findall(pattern, text)

        image_files = []
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
                        msg = f"Invalid max_used_frames value: {max_used_frames}. Expected 1-{max_frames_spacing}"
                        raise ValueError(msg)
                buffer, ext = encode_media_frames(img_)
                if len(buffer) > 1:
                    step = max(1, len(buffer) // max_frames_spacing)
                    indices = list(range(0, len(buffer), step))[:max_used_frames]
                    for i in indices:
                        image_files.append(f"data:image/{ext};base64,{buffer[i]}")
                elif len(buffer) == 1:
                    image_files.append(f"data:image/{ext};base64,{buffer[0]}")
                else:
                    logger.warning("No frames found or error in encoding frames")
        return image_files

    def _remove_vision_pattern(self, text: str) -> str:
        # This regular expression matches <<vision:...:>> patterns to strip them from output text.
        pattern = r"<<vision:(.*?):>>"
        return re.sub(pattern, "", text)

    def _build_system_content(self, argument, image_files: list[str]) -> str:
        sections = []
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
        sections = []
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
        key = "developer" if self.is_openai_reasoning_model() else "system"
        res = self_prompter({"user": user_text, key: system})
        if res is None:
            msg = "Self-prompting failed!"
            raise ValueError(msg)
        new_user_msg = self._create_user_message(res["user"], image_files)
        return res[key], new_user_msg

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

        image_files = self._handle_image_content(str(argument.prop.processed_input))
        if image_files and not self.supports_openai_vision():
            msg = f"Model {self.model} does not support vision input."
            raise ValueError(msg)
        system_content = self._build_system_content(argument, image_files)
        user_text = self._build_user_text(argument, image_files)
        user_msg = self._create_user_message(user_text, image_files)
        system_content, user_msg = self._apply_self_prompt_if_needed(
            argument, system_content, user_msg, user_text, image_files
        )

        role = "developer" if self.is_openai_reasoning_model() else "system"
        argument.prop.prepared_input = [
            {"role": role, "content": system_content},
            user_msg,
        ]

    def build_request(self, argument) -> CreateResponseRequest:
        unsupported = {
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
            "max_tool_calls",
            "parallel_tool_calls",
        } & set(argument.kwargs)
        if unsupported:
            msg = (
                "OpenAI Responses integration does not support tool calling or streaming: "
                f"{sorted(unsupported)}"
            )
            raise ValueError(msg)

        if argument.kwargs.get("background") is True:
            msg = "OpenAIResponsesEngine does not support background execution."
            raise ValueError(msg)

        request_kwargs = set(CreateResponseRequest.model_fields) - {"input"}
        payload = self.collect_request_kwargs(argument, request_kwargs)
        model = payload.get("model", self.model)
        self.openai_model_spec_for(model)
        payload["model"] = model
        payload["input"] = tuple(
            TypeAdapter(InputMessage).validate_python(message, strict=False)
            for message in argument.prop.prepared_input
        )

        if self.is_openai_reasoning_model():
            payload.pop("temperature", None)
            payload.pop("top_p", None)
            default_effort = "high" if self.is_openai_pro_model() else "medium"
            payload["reasoning"] = payload.get(
                "reasoning",
                {"effort": default_effort},
            )

        converters = {
            "conversation": Conversation,
            "moderation": ModerationConfig,
            "prompt": PromptReference,
            "prompt_cache_options": PromptCacheOptions,
            "reasoning": ReasoningConfig,
            "text": TextConfig,
        }
        for field_name, model_type in converters.items():
            if isinstance(payload.get(field_name), dict):
                payload[field_name] = model_type.model_validate(
                    payload[field_name],
                    strict=False,
                )

        if isinstance(payload.get("context_management"), list):
            payload["context_management"] = tuple(
                ContextCompaction.model_validate(item, strict=False)
                for item in payload["context_management"]
            )
        if isinstance(payload.get("include"), list):
            payload["include"] = tuple(payload["include"])
        if isinstance(payload.get("instructions"), list):
            payload["instructions"] = tuple(
                TypeAdapter(InputMessage).validate_python(item, strict=False)
                for item in payload["instructions"]
            )
        if isinstance(payload.get("service_tier"), str):
            payload["service_tier"] = ServiceTier(payload["service_tier"])
        if isinstance(payload.get("truncation"), str):
            payload["truncation"] = Truncation(payload["truncation"])

        request = CreateResponseRequest.model_validate(payload)
        remaining_tokens = self.compute_remaining_tokens(argument.prop.prepared_input)
        max_output_tokens = request.max_output_tokens
        if max_output_tokens is not None and max_output_tokens > self.max_response_tokens:
            warning_message = (
                f"Provided 'max_output_tokens' ({max_output_tokens}) exceeds max "
                f"({self.max_response_tokens}). Truncating to {remaining_tokens}."
            )
            logger.warning(warning_message)
            request = request.model_copy(update={"max_output_tokens": remaining_tokens})
        return request

    def _extract_output_text(self, response: Response) -> list[str]:
        outputs = []
        for output in response.output:
            if not isinstance(output, OutputMessage):
                continue
            outputs.extend(
                content.text for content in output.content if isinstance(content, OutputText)
            )
        return outputs

    def _extract_thinking(self, response: Response) -> str | None:
        if not self.is_openai_reasoning_model():
            return None
        for output in response.output:
            if output.type == "reasoning":
                texts = [summary.text for summary in output.summary]
                if texts:
                    return "\n".join(texts)
        return None

    def forward(self, argument):  # pyright: ignore[reportIncompatibleMethodOverride]
        if not self.api_key:
            msg = "OpenAI API key is not set."
            raise ValueError(msg)
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def call_request(self, request: CreateResponseRequest) -> APIResponse[Response]:
        if self.http_client is not None:
            return OpenAIClient(
                api_key=self.api_key,
                http_client=self.http_client,
            ).create_response(request)

        timeout = self.client_timeout if self.client_timeout is not None else 600.0
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0)) as http_client:
            return OpenAIClient(
                api_key=self.api_key,
                http_client=http_client,
            ).create_response(request)

    def parse_response(self, response: APIResponse[Response]):
        raw_output = response.data
        if raw_output.status is not ResponseStatus.COMPLETED:
            detail = f": {raw_output.error.message}" if raw_output.error else ""
            msg = (
                "OpenAIResponsesEngine requires a completed response; "
                f"received status {raw_output.status.value!r}{detail}"
            )
            raise ValueError(msg)

        metadata = {
            "raw_output": raw_output,
            "response": response,
        }

        thinking = self._extract_thinking(raw_output)
        if thinking:
            metadata["thinking"] = thinking

        return self._extract_output_text(raw_output), metadata
