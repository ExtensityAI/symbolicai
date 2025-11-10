import json
import logging
import re
from copy import deepcopy

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


class GPTXReasoningEngine(Engine, OpenAIMixin):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        # In case we use EngineRepository.register to inject the api_key and model => dynamically change the engine at runtime
        if api_key is not None and model is not None:
            self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] = api_key
            self.config['NEUROSYMBOLIC_ENGINE_MODEL']   = model
        if self.id() != 'neurosymbolic':
            return # do not initialize if not neurosymbolic; avoids conflict with llama.cpp check in EngineRepository.register_from_package
        openai.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model = self.config['NEUROSYMBOLIC_ENGINE_MODEL']
        self.name = self.__class__.__name__
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding('o200k_base')
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.seed = None

        try:
            self.client = openai.Client(api_key=openai.api_key)
        except Exception as e:
            UserMessage(f'Failed to initialize OpenAI client. Please check your OpenAI library version. Caused by: {e}', raise_with=ValueError)

    def id(self) -> str:
        if self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
           (self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('o1') or \
            self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('o3') or \
            self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('o4') or \
            self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') == 'gpt-5' or \
            self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') == 'gpt-5-mini' or \
            self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') == 'gpt-5-nano'):
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

    def compute_required_tokens(self, messages):
        """Return the number of tokens used by a list of messages."""

        if self.model in {
            'o1',
            'o3',
            'o3-mini',
            'o4-mini',
            'gpt-5',
            'gpt-5-mini',
            'gpt-5-nano',
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            UserMessage(
                f"'num_tokens_from_messages()' is not implemented for model {self.model}. "
                "See https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken for information on how messages are converted to tokens.",
                raise_with=NotImplementedError
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(self.tokenizer.encode(value, disallowed_special=()))
                else:
                    for v in value:
                        if v['type'] == 'text':
                            num_tokens += len(self.tokenizer.encode(v['text'], disallowed_special=()))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens - 1 # don't know where that extra 1 comes from

    def compute_remaining_tokens(self, prompts: list) -> int:
        val = self.compute_required_tokens(prompts)
        return min(self.max_context_tokens - val, self.max_response_tokens)

    def _handle_image_content(self, content: str) -> list:
        """Handle image content by processing vision patterns and returning image file data."""
        def _extract_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.findall(pattern, text)

        image_files = []
        # pre-process prompt if contains image url
        if (self.model == 'o1' or \
            self.model == 'gpt-5' or \
            self.model == 'gpt-5-mini' or \
            self.model == 'gpt-5-nano') and '<<vision:' in content:
            parts = _extract_pattern(content)
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

    def _slice_tokens(self, tokens, new_len, truncation_type):
        """Slice tokens based on truncation type."""
        new_len = max(100, new_len)  # Ensure minimum token length
        return tokens[-new_len:] if truncation_type == 'head' else tokens[:new_len]  # else 'tail'

    def _validate_truncation_prompts(self, prompts: list[dict]) -> bool:
        """Validate prompt structure before truncation."""
        if len(prompts) != 2 and all(prompt['role'] in ['developer', 'user'] for prompt in prompts):
            # Only support developer and user prompts
            UserMessage(
                f"Token truncation currently supports only two messages, from 'user' and 'developer' (got {len(prompts)}). Returning original prompts."
            )
            return False
        return True

    def _collect_user_tokens(
        self,
        user_prompt: dict,
    ) -> tuple[list[int], bool]:
        """Collect user tokens and detect unsupported content."""
        user_tokens: list[int] = []
        user_content = user_prompt['content']
        if isinstance(user_content, str):
            user_tokens.extend(Symbol(user_content).tokens)
            return user_tokens, False
        if isinstance(user_content, list):
            for content_item in user_content:
                if isinstance(content_item, dict):
                    if content_item.get('type') == 'text':
                        user_tokens.extend(Symbol(content_item['text']).tokens)
                    else:
                        return user_tokens, True
                else:
                    UserMessage(
                        f"Invalid content type: {type(content_item)}. Format input according to the documentation. See https://platform.openai.com/docs/api-reference/chat/create?lang=python",
                        raise_with=ValueError,
                    )
            return user_tokens, False
        return UserMessage(
            f"Unknown content type: {type(user_prompt['content'])}. Format input according to the documentation. See https://platform.openai.com/docs/api-reference/chat/create?lang=python",
            raise_with=ValueError,
        )

    def _truncate_single_prompt_exceed(
        self,
        system_tokens,
        user_tokens,
        system_token_count,
        user_token_count,
        max_prompt_tokens,
        truncation_type,
    ):
        """Handle truncation when only one prompt exceeds the limit."""
        half_limit = max_prompt_tokens / 2
        if user_token_count > half_limit and system_token_count <= half_limit:
            new_user_len = max_prompt_tokens - system_token_count
            new_user_tokens = self._slice_tokens(user_tokens, new_user_len, truncation_type)
            return [
                {'role': 'developer', 'content': self.tokenizer.decode(system_tokens)},
                {'role': 'user', 'content': [{'type': 'text', 'text': self.tokenizer.decode(new_user_tokens)}]},
            ]
        if system_token_count > half_limit and user_token_count <= half_limit:
            new_system_len = max_prompt_tokens - user_token_count
            new_system_tokens = self._slice_tokens(system_tokens, new_system_len, truncation_type)
            return [
                {'role': 'developer', 'content': self.tokenizer.decode(new_system_tokens)},
                {'role': 'user', 'content': [{'type': 'text', 'text': self.tokenizer.decode(user_tokens)}]},
            ]
        return None

    def truncate(self, prompts: list[dict], truncation_percentage: float | None, truncation_type: str) -> list[dict]:
        """Main truncation method"""
        if not self._validate_truncation_prompts(prompts):
            return prompts

        if truncation_percentage is None:
            # Calculate smart truncation percentage based on model's max messages and completion tokens
            truncation_percentage = (self.max_context_tokens - 16_384) / self.max_context_tokens

        system_prompt = prompts[0]
        user_prompt = prompts[1]

        # Get token counts
        system_tokens = Symbol(system_prompt['content']).tokens
        user_tokens = []

        user_tokens, should_return_original = self._collect_user_tokens(user_prompt)
        if should_return_original:
            return prompts

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

        single_prompt_adjustment = self._truncate_single_prompt_exceed(
            system_tokens,
            user_tokens,
            system_token_count,
            user_token_count,
            max_prompt_tokens,
            truncation_type,
        )
        if single_prompt_adjustment is not None:
            return single_prompt_adjustment

        # Case 3: Both exceed - reduce proportionally
        system_ratio = system_token_count / total_tokens
        user_ratio = user_token_count / total_tokens

        new_system_len = int(max_prompt_tokens * system_ratio)
        new_user_len = int(max_prompt_tokens * user_ratio)
        distribute_tokens = max_prompt_tokens - new_system_len - new_user_len
        new_system_len += distribute_tokens // 2
        new_user_len += distribute_tokens // 2

        new_system_tokens = self._slice_tokens(system_tokens, new_system_len, truncation_type)
        new_user_tokens = self._slice_tokens(user_tokens, new_user_len, truncation_type)

        return [
            {'role': 'developer', 'content': self.tokenizer.decode(new_system_tokens)},
            {'role': 'user', 'content': [{'type': 'text', 'text': self.tokenizer.decode(new_user_tokens)}]}
        ]

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

    def _non_verbose_section(self, argument) -> str:
        """Return non-verbose instruction section if needed."""
        if argument.prop.suppress_verbose_output:
            return (
                "<META_INSTRUCTION/>\n"
                "You do not output anything else, like verbose preambles or post explanation, such as "
                "\"Sure, let me...\", \"Hope that was helpful...\", \"Yes, I can help you with that...\", etc. "
                "Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use "
                "indentation, etc. Never add meta instructions information to your output!\n\n"
            )
        return ''

    def _response_format_section(self, argument) -> str:
        """Return response format instructions if provided."""
        if not argument.prop.response_format:
            return ''
        response_format = argument.prop.response_format
        assert response_format.get('type') is not None, 'Expected format `{ "type": "json_object" }`! See https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format'
        if response_format["type"] == "json_object":
            return '<RESPONSE_FORMAT/>\nYou are a helpful assistant designed to output JSON.\n\n'
        return ''

    def _context_sections(self, argument) -> list[str]:
        """Return static and dynamic context sections."""
        sections: list[str] = []
        static_ctxt, dyn_ctxt = argument.prop.instance.global_context
        if len(static_ctxt) > 0:
            sections.append(f"<STATIC CONTEXT/>\n{static_ctxt}\n\n")
        if len(dyn_ctxt) > 0:
            sections.append(f"<DYNAMIC CONTEXT/>\n{dyn_ctxt}\n\n")
        return sections

    def _additional_context_section(self, argument) -> str:
        """Return additional payload context if any."""
        if argument.prop.payload:
            return f"<ADDITIONAL CONTEXT/>\n{argument.prop.payload!s}\n\n"
        return ''

    def _examples_section(self, argument) -> str:
        """Return examples section if provided."""
        examples: list[str] = argument.prop.examples
        if examples and len(examples) > 0:
            return f"<EXAMPLES/>\n{examples!s}\n\n"
        return ''

    def _instruction_section(self, argument, image_files: list[str]) -> str:
        """Return instruction section, removing vision patterns when needed."""
        prompt = argument.prop.prompt
        if prompt is None or len(prompt) == 0:
            return ''
        value = str(prompt)
        if len(image_files) > 0:
            value = self._remove_vision_pattern(value)
        return f"<INSTRUCTION/>\n{value}\n\n"

    def _build_developer_prompt(self, argument, image_files: list[str]) -> str:
        """Assemble developer prompt content."""
        developer = self._non_verbose_section(argument)
        developer = f'{developer}\n' if developer else ''

        parts = [
            self._response_format_section(argument),
            *self._context_sections(argument),
            self._additional_context_section(argument),
            self._examples_section(argument),
            self._instruction_section(argument, image_files),
        ]
        developer += ''.join(part for part in parts if part)

        if argument.prop.template_suffix:
            developer += (
                f' You will only generate content for the placeholder `{argument.prop.template_suffix!s}` '
                'following the instructions and the provided context information.\n\n'
            )
        return developer

    def _build_user_suffix(self, argument, image_files: list[str]) -> str:
        """Prepare user content suffix."""
        suffix: str = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = self._remove_vision_pattern(suffix)
        return suffix

    def _construct_user_prompt(self, user_text: str, image_files: list[str]):
        """Construct user prompt payload."""
        if self.model in {
                'o1',
                'o3',
                'o3-mini',
                'o4-mini',
                'gpt-5',
                'gpt-5-mini',
                'gpt-5-nano',
            }:
            images = [{'type': 'image_url', 'image_url': {'url': file}} for file in image_files]
            user_prompt = {
                "role": "user",
                "content": [
                    *images,
                    {'type': 'text', 'text': user_text},
                ],
            }
            return user_prompt, images
        return {"role": "user", "content": user_text}, None

    def _apply_self_prompt(
        self,
        argument,
        user_prompt,
        developer: str,
        user_text: str,
        images,
        image_files: list[str],
    ):
        """Apply self-prompting when requested."""
        instance = argument.prop.instance
        if not (instance._kwargs.get('self_prompt', False) or argument.prop.self_prompt):
            return user_prompt, developer

        self_prompter = SelfPrompt()
        res = self_prompter({'user': user_text, 'developer': developer})
        if res is None:
            UserMessage("Self-prompting failed!", raise_with=ValueError)

        if len(image_files) > 0:
            image_content = images if images is not None else [
                {'type': 'image_url', 'image_url': {'url': file}} for file in image_files
            ]
            user_prompt = {
                "role": "user",
                "content": [
                    *image_content,
                    {'type': 'text', 'text': res['user']},
                ],
            }
        else:
            user_prompt = {"role": "user", "content": res['user']}

        return user_prompt, res['developer']

    def prepare(self, argument):
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        image_files = self._handle_image_content(str(argument.prop.processed_input))

        developer = self._build_developer_prompt(argument, image_files)
        user_text = self._build_user_suffix(argument, image_files)
        user_prompt, images = self._construct_user_prompt(user_text, image_files)
        user_prompt, developer = self._apply_self_prompt(
            argument,
            user_prompt,
            developer,
            user_text,
            images,
            image_files,
        )

        argument.prop.prepared_input = [
            { "role": "developer", "content": developer },
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
                if hit:
                    UserMessage("Multiple function calls detected in the response but only the first one will be processed.")
                    break
                if hasattr(tool_call, 'function') and tool_call.function:
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
            "reasoning_effort": kwargs.get('reasoning_effort', 'medium'),
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
        }

        if self.model == "o4-mini" or self.model == "o3":
            del payload["stop"]

        if self.model.startswith("gpt-5"):
            del payload["stop"]

        return payload
