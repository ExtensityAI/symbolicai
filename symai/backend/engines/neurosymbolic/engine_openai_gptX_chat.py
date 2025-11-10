import json
import logging
import re
from copy import deepcopy
from typing import ClassVar

import openai
import tiktoken

from ....components import SelfPrompt
from ....symbol import Symbol
from ....utils import UserMessage, encode_media_frames
from ...base import Engine
from ...mixin.openai import OpenAIMixin
from ...settings import SYMAI_CONFIG

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class GPTXChatEngine(Engine, OpenAIMixin):
    _THREE_TOKEN_MODELS: ClassVar[set[str]] = {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-1106-preview",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini",
        "chatgpt-4o-latest",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5-chat-latest",
    }
    _VISION_PREVIEW_MODEL = "gpt-4-vision-preview"
    _VISION_IMAGE_URL_MODELS: ClassVar[set[str]] = {
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "chatgpt-4o-latest",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5-chat-latest",
    }
    _NON_VERBOSE_OUTPUT = """<META_INSTRUCTION/>\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n\n"""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        # In case we use EngineRepository.register to inject the api_key and model => dynamically change the engine at runtime
        if api_key is not None and model is not None:
            self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] = api_key
            self.config['NEUROSYMBOLIC_ENGINE_MODEL'] = model
        if self.id() != 'neurosymbolic':
            return # do not initialize if not neurosymbolic; avoids conflict with llama.cpp check in EngineRepository.register_from_package
        openai.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model = self.config['NEUROSYMBOLIC_ENGINE_MODEL']
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding('o200k_base')
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.seed = None
        self.name = self.__class__.__name__

        try:
            self.client = openai.Client(api_key=openai.api_key)
        except Exception as e:
            UserMessage(f'Failed to initialize OpenAI client. Please check your OpenAI library version. Caused by: {e}', raise_with=ValueError)

    def id(self) -> str:
        if self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
           (self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('gpt-3.5') or \
            self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('gpt-4') or \
            self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('chatgpt-4o') or \
            self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('gpt-4.1') or \
            self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') == 'gpt-5-chat-latest'):
                return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            openai.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']
        if 'seed' in kwargs:
            self.seed = kwargs['seed']

    def _resolve_token_config(self) -> tuple[int, int]:
        if self.model in self._THREE_TOKEN_MODELS:
            return 3, 1
        if self.model == "gpt-3.5-turbo-0301":
            return 4, -1
        if self.model == "gpt-3.5-turbo":
            UserMessage("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
            return 3, 1
        if self.model == "gpt-4":
            UserMessage("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            self.tokenizer = tiktoken.encoding_for_model("gpt-4-0613")
            return 3, 1
        UserMessage(
            f"""num_tokens_from_messages() is not implemented for model {self.model}. See https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken for information on how messages are converted to tokens.""",
            raise_with=NotImplementedError
        )
        raise NotImplementedError

    def _count_tokens_in_value(self, value) -> int:
        if isinstance(value, str):
            return len(self.tokenizer.encode(value, disallowed_special=()))
        tokens = 0
        for item in value:
            if item['type'] == 'text':
                tokens += len(self.tokenizer.encode(item['text'], disallowed_special=()))
        return tokens

    def compute_required_tokens(self, messages):
        """Return the number of tokens used by a list of messages."""

        tokens_per_message, tokens_per_name = self._resolve_token_config()
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += self._count_tokens_in_value(value)
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def compute_remaining_tokens(self, prompts: list) -> int:
        val = self.compute_required_tokens(prompts)
        return min(self.max_context_tokens - val, self.max_response_tokens)

    def _should_skip_truncation(self, prompts: list[dict]) -> bool:
        if len(prompts) != 2 and all(prompt['role'] in ['system', 'user'] for prompt in prompts):
            UserMessage(f"Token truncation currently supports only two messages, from 'user' and 'system' (got {len(prompts)}). Returning original prompts.")
            return True
        return False

    def _resolve_truncation_percentage(self, truncation_percentage: float | None) -> float:
        if truncation_percentage is not None:
            return truncation_percentage
        return (self.max_context_tokens - self.max_response_tokens) / self.max_context_tokens

    def _collect_user_tokens(self, user_prompt: dict, prompts: list[dict]) -> tuple[list, object | None]:
        user_tokens: list = []
        content = user_prompt['content']
        if isinstance(content, str):
            user_tokens.extend(Symbol(content).tokens)
            return user_tokens, None
        if isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict):
                    if content_item.get('type') == 'text':
                        user_tokens.extend(Symbol(content_item['text']).tokens)
                    else:
                        return [], prompts
                else:
                    return [], ValueError(f"Invalid content type: {type(content_item)}. Format input according to the documentation. See https://platform.openai.com/docs/api-reference/chat/create?lang=python")
            return user_tokens, None
        UserMessage(f"Unknown content type: {type(content)}. Format input according to the documentation. See https://platform.openai.com/docs/api-reference/chat/create?lang=python", raise_with=ValueError)
        return user_tokens, None

    def _user_only_exceeds(self, user_token_count: int, system_token_count: int, max_prompt_tokens: int) -> bool:
        return user_token_count > max_prompt_tokens/2 and system_token_count <= max_prompt_tokens/2

    def _system_only_exceeds(self, system_token_count: int, user_token_count: int, max_prompt_tokens: int) -> bool:
        return system_token_count > max_prompt_tokens/2 and user_token_count <= max_prompt_tokens/2

    def _compute_proportional_lengths(self, system_token_count: int, user_token_count: int, total_tokens: int, max_prompt_tokens: int) -> tuple[int, int]:
        system_ratio = system_token_count / total_tokens
        user_ratio = user_token_count / total_tokens
        new_system_len = int(max_prompt_tokens * system_ratio)
        new_user_len = int(max_prompt_tokens * user_ratio)
        distribute_tokens = max_prompt_tokens - new_system_len - new_user_len
        new_system_len += distribute_tokens // 2
        new_user_len += distribute_tokens // 2
        return new_system_len, new_user_len

    def _decode_prompt_pair(self, system_tokens, user_tokens) -> list[dict]:
        return [
            {'role': 'system', 'content': self.tokenizer.decode(system_tokens)},
            {'role': 'user', 'content': [{'type': 'text', 'text': self.tokenizer.decode(user_tokens)}]}
        ]

    def _handle_image_content(self, content: str) -> list:
        """Handle image content by processing vision patterns and returning image file data."""
        def extract_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.findall(pattern, text)

        image_files = []
        # pre-process prompt if contains image url
        if (self.model == 'gpt-4-vision-preview' or \
            self.model == 'gpt-4-turbo-2024-04-09' or \
            self.model == 'gpt-4-turbo' or \
            self.model == 'gpt-4o' or \
            self.model == 'gpt-4o-mini' or \
            self.model == 'chatgpt-4o-latest' or \
            self.model == 'gpt-4.1' or \
            self.model == 'gpt-4.1-mini' or \
            self.model == 'gpt-4.1-nano' or \
            self.model == 'gpt-5-chat-latest') \
            and '<<vision:' in content:

            parts = extract_pattern(content)
            for p in parts:
                img_ = p.strip()
                if img_.startswith('http') or img_.startswith('data:image'):
                    image_files.append(img_)
                else:
                    max_frames_spacing = 50
                    max_used_frames = 10
                    if img_.startswith('frames:'):
                        img_ = img_.replace('frames:', '')
                        max_used_frames, img_ = img_.split(':')
                        max_used_frames = int(max_used_frames)
                        if max_used_frames < 1 or max_used_frames > max_frames_spacing:
                            UserMessage(f"Invalid max_used_frames value: {max_used_frames}. Expected value between 1 and {max_frames_spacing}", raise_with=ValueError)
                    buffer, ext = encode_media_frames(img_)
                    if len(buffer) > 1:
                        step = len(buffer) // max_frames_spacing # max frames spacing
                        frames = []
                        indices = list(range(0, len(buffer), step))[:max_used_frames]
                        for i in indices:
                            frames.append(f"data:image/{ext};base64,{buffer[i]}")
                        image_files.extend(frames)
                    elif len(buffer) == 1:
                        image_files.append(f"data:image/{ext};base64,{buffer[0]}")
                    else:
                        UserMessage('No frames found or error in encoding frames')
        return image_files

    def _remove_vision_pattern(self, text: str) -> str:
        """Remove vision patterns from text."""
        pattern = r'<<vision:(.*?):>>'
        return re.sub(pattern, '', text)

    def truncate(self, prompts: list[dict], truncation_percentage: float | None, truncation_type: str) -> list[dict]:
        """Main truncation method"""
        def _slice_tokens(tokens, new_len, truncation_type):
            """Slice tokens based on truncation type"""
            new_len = max(100, new_len)  # Ensure minimum token length
            return tokens[-new_len:] if truncation_type == 'head' else tokens[:new_len] # else 'tail'

        if self._should_skip_truncation(prompts):
            return prompts

        truncation_percentage = self._resolve_truncation_percentage(truncation_percentage)
        system_prompt = prompts[0]
        user_prompt = prompts[1]
        system_tokens = Symbol(system_prompt['content']).tokens
        user_tokens, fallback = self._collect_user_tokens(user_prompt, prompts)
        if fallback is not None:
            return fallback
        system_token_count = len(system_tokens)
        user_token_count = len(user_tokens)
        artifacts = self.compute_required_tokens(prompts) - (system_token_count + user_token_count)
        assert artifacts >= 0, f"Artifacts count is negative: {artifacts}! Report bug!"
        total_tokens = system_token_count + user_token_count + artifacts

        # Calculate maximum allowed tokens
        max_prompt_tokens = int(self.max_context_tokens * truncation_percentage)

        # If total is within limit, return original
        if total_tokens <= max_prompt_tokens:
            return prompts

        UserMessage(
            f"Executing {truncation_type} truncation to fit within {max_prompt_tokens} tokens. "
            f"Combined prompts ({total_tokens} tokens) exceed maximum allowed tokens "
            f"of {max_prompt_tokens} ({truncation_percentage*100:.1f}% of context). "
            f"You can control this behavior by setting 'truncation_percentage' (current: {truncation_percentage:.2f}) "
            f"and 'truncation_type' (current: '{truncation_type}') parameters. "
            f"Set 'truncation_percentage=1.0' to deactivate truncation (will fail if exceeding context window). "
            f"Choose 'truncation_type' as 'head' to keep the end of prompts or 'tail' to keep the beginning."
        )
        # Case 1: Only user prompt exceeds
        if self._user_only_exceeds(user_token_count, system_token_count, max_prompt_tokens):
            new_user_len = max_prompt_tokens - system_token_count
            new_user_tokens = _slice_tokens(user_tokens, new_user_len, truncation_type)
            return self._decode_prompt_pair(system_tokens, new_user_tokens)

        # Case 2: Only system prompt exceeds
        if self._system_only_exceeds(system_token_count, user_token_count, max_prompt_tokens):
            new_system_len = max_prompt_tokens - user_token_count
            new_system_tokens = _slice_tokens(system_tokens, new_system_len, truncation_type)
            return self._decode_prompt_pair(new_system_tokens, user_tokens)

        # Case 3: Both exceed - reduce proportionally
        new_system_len, new_user_len = self._compute_proportional_lengths(system_token_count, user_token_count, total_tokens, max_prompt_tokens)
        new_system_tokens = _slice_tokens(system_tokens, new_system_len, truncation_type)
        new_user_tokens = _slice_tokens(user_tokens, new_user_len, truncation_type)

        return self._decode_prompt_pair(new_system_tokens, new_user_tokens)

    def forward(self, argument):
        kwargs = argument.kwargs
        truncation_percentage = kwargs.get('truncation_percentage', argument.prop.truncation_percentage)
        truncation_type = kwargs.get('truncation_type', argument.prop.truncation_type)
        messages = self.truncate(argument.prop.prepared_input, truncation_percentage, truncation_type)
        payload = self._prepare_request_payload(messages, argument)
        except_remedy = kwargs.get('except_remedy')

        try:
            res = self.client.chat.completions.create(**payload)

        except Exception as e:
            if openai.api_key is None or openai.api_key == '':
                msg = 'OpenAI API key is not set. Please set it in the config file or pass it as an argument to the command method.'
                UserMessage(msg)
                if self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] == '':
                    UserMessage(msg, raise_with=ValueError)
                openai.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']

            callback = self.client.chat.completions.create
            kwargs['model'] = kwargs.get('model', self.model)

            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                UserMessage(f'Error during generation. Caused by: {e}', raise_with=ValueError)

        metadata = {'raw_output': res}
        if payload.get('tools'):
            metadata = self._process_function_calls(res, metadata)
        output = [r.message.content for r in res.choices]

        #@TODO: Normalize the output across engines to result something like Result object
        #       I like the Rust Ok Result object, there's something similar in Python
        #       (https://github.com/rustedpy/result)
        return output, metadata

    def _prepare_raw_input(self, argument):
        if not argument.prop.processed_input:
            UserMessage('Need to provide a prompt instruction to the engine if raw_input is enabled.', raise_with=ValueError)
        value = argument.prop.processed_input
        # convert to dict if not already
        if not isinstance(value, list):
            if not isinstance(value, dict):
                value = {'role': 'user', 'content': str(value)}
            value = [value]
        return value

    def _build_non_verbose_prefix(self, argument) -> list[str]:
        if not argument.prop.suppress_verbose_output:
            return []
        prefix = f'{self._NON_VERBOSE_OUTPUT}\n'
        return [prefix]

    def _response_format_section(self, argument) -> list[str]:
        if not argument.prop.response_format:
            return []
        _rsp_fmt = argument.prop.response_format
        assert _rsp_fmt.get('type') is not None, 'Expected format `{ "type": "json_object" }`! See https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format'
        if _rsp_fmt["type"] != "json_object":
            return []
        return ['<RESPONSE_FORMAT/>\nYou are a helpful assistant designed to output JSON.\n\n']

    def _context_sections(self, argument) -> list[str]:
        sections: list[str] = []
        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            sections.append(f"<STATIC CONTEXT/>\n{static_ctxt}\n\n")
        if len(dyn_ctxt) > 0:
            sections.append(f"<DYNAMIC CONTEXT/>\n{dyn_ctxt}\n\n")
        return sections

    def _payload_section(self, argument) -> list[str]:
        if not argument.prop.payload:
            return []
        payload = argument.prop.payload
        return [f"<ADDITIONAL CONTEXT/>\n{payload!s}\n\n"]

    def _examples_section(self, argument) -> list[str]:
        examples: list[str] = argument.prop.examples
        if not (examples and len(examples) > 0):
            return []
        return [f"<EXAMPLES/>\n{examples!s}\n\n"]

    def _instruction_section(self, argument, image_files: list[str]) -> list[str]:
        if argument.prop.prompt is None or len(argument.prop.prompt) == 0:
            return []
        val = str(argument.prop.prompt)
        if len(image_files) > 0:
            val = self._remove_vision_pattern(val)
        return [f"<INSTRUCTION/>\n{val}\n\n"]

    def _template_suffix_section(self, argument) -> list[str]:
        if not argument.prop.template_suffix:
            return []
        return [f' You will only generate content for the placeholder `{argument.prop.template_suffix!s}` following the instructions and the provided context information.\n\n']

    def _build_system_message(self, argument, image_files: list[str]) -> str:
        sections: list[str] = []
        sections.extend(self._build_non_verbose_prefix(argument))
        sections.extend(self._response_format_section(argument))
        sections.extend(self._context_sections(argument))
        sections.extend(self._payload_section(argument))
        sections.extend(self._examples_section(argument))
        sections.extend(self._instruction_section(argument, image_files))
        sections.extend(self._template_suffix_section(argument))
        return "".join(sections)

    def _build_user_text(self, argument, image_files: list[str]) -> str:
        suffix: str = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = self._remove_vision_pattern(suffix)
        return f"{suffix}"

    def _create_user_prompt(self, user_text: str, image_files: list[str]) -> dict:
        if self.model == self._VISION_PREVIEW_MODEL:
            images = [{'type': 'image', "image_url": {"url": file}} for file in image_files]
            return {"role": "user", "content": [*images, {'type': 'text', 'text': user_text}]}
        if self.model in self._VISION_IMAGE_URL_MODELS:
            images = [{'type': 'image_url', "image_url": {"url": file}} for file in image_files]
            return {"role": "user", "content": [*images, {'type': 'text', 'text': user_text}]}
        return {"role": "user", "content": user_text}

    def _apply_self_prompt_if_needed(self, argument, system: str, user_prompt: dict, user_text: str, image_files: list[str]) -> tuple[str, dict]:
        if not (argument.prop.instance._kwargs.get('self_prompt', False) or argument.prop.self_prompt):
            return system, user_prompt
        self_prompter = SelfPrompt()
        res = self_prompter({'user': user_text, 'system': system})
        if res is None:
            UserMessage("Self-prompting failed!", raise_with=ValueError)
        new_user_prompt = self._create_user_prompt(res['user'], image_files)
        return res['system'], new_user_prompt

    def prepare(self, argument):
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        image_files = self._handle_image_content(str(argument.prop.processed_input))
        system = self._build_system_message(argument, image_files)
        user_text = self._build_user_text(argument, image_files)
        user_prompt = self._create_user_prompt(user_text, image_files)
        system, user_prompt = self._apply_self_prompt_if_needed(argument, system, user_prompt, user_text, image_files)

        argument.prop.prepared_input = [
            { "role": "system", "content": system },
            user_prompt,
        ]

    def _process_function_calls(self, res, metadata):
        hit = False
        if (
            hasattr(res, 'choices')
            and res.choices
            and hasattr(res.choices[0], 'message')
            and res.choices[0].message
            and hasattr(res.choices[0].message, 'tool_calls')
            and res.choices[0].message.tool_calls
        ):
            for tool_call in res.choices[0].message.tool_calls:
                if hasattr(tool_call, 'function') and tool_call.function:
                    if hit:
                        UserMessage("Multiple function calls detected in the response but only the first one will be processed.")
                        break
                    try:
                        args_dict = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args_dict = {}
                    metadata['function_call'] = {
                        'name': tool_call.function.name,
                        'arguments': args_dict
                    }
                    hit = True
        return metadata

    def _prepare_request_payload(self, messages, argument):
        """Prepares the request payload from the argument."""
        kwargs = argument.kwargs

        max_tokens = kwargs.get('max_tokens', None)
        max_completion_tokens = kwargs.get('max_completion_tokens', None)
        remaining_tokens = self.compute_remaining_tokens(messages)

        if max_tokens is not None:
            UserMessage(
                "'max_tokens' is now deprecated in favor of 'max_completion_tokens', and is not compatible with o1 series models. "
                "We handle this conversion by default for you for now but we won't in the future. "
                "See: https://platform.openai.com/docs/api-reference/chat/create"
            )
            if max_tokens > self.max_response_tokens:
                UserMessage(
                    f"Provided 'max_tokens' ({max_tokens}) exceeds max response tokens ({self.max_response_tokens}). "
                    f"Truncating to {remaining_tokens} to avoid API failure."
                )
                kwargs['max_completion_tokens'] = remaining_tokens
            else:
                kwargs['max_completion_tokens'] = max_tokens
            del kwargs['max_tokens']

        if max_completion_tokens is not None and max_completion_tokens > self.max_response_tokens:
            UserMessage(
                f"Provided 'max_completion_tokens' ({max_completion_tokens}) exceeds max response tokens ({self.max_response_tokens}). "
                f"Truncating to {remaining_tokens} to avoid API failure."
            )
            kwargs['max_completion_tokens'] = remaining_tokens

        payload = {
            "messages": messages,
            "model": kwargs.get('model', self.model),
            "seed": kwargs.get('seed', self.seed),
            "max_completion_tokens": kwargs.get('max_completion_tokens'),
            "stop": kwargs.get('stop', ''),
            "temperature": kwargs.get('temperature', 1),
            "frequency_penalty": kwargs.get('frequency_penalty', 0),
            "presence_penalty": kwargs.get('presence_penalty', 0),
            "top_p": kwargs.get('top_p', 1),
            "n": kwargs.get('n', 1),
            "logit_bias": kwargs.get('logit_bias'),
            "tools": kwargs.get('tools'),
            "tool_choice": kwargs.get('tool_choice'),
            "response_format": kwargs.get('response_format'),
            "logprobs": kwargs.get('logprobs'),
            "top_logprobs": kwargs.get('top_logprobs'),
        }

        if self.model == "chatgpt-4o-latest" or self.model == "gpt-5-chat-latest":
            del payload['tools']
            del payload['tool_choice']

        return payload
