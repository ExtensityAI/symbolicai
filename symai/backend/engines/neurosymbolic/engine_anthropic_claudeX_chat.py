import json
import logging
import re
from copy import copy, deepcopy

import anthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import (
    InputJSONDelta,
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)

from ....components import SelfPrompt
from ....utils import UserMessage, encode_media_frames
from ...base import Engine
from ...mixin.anthropic import AnthropicMixin
from ...settings import SYMAI_CONFIG

logging.getLogger("anthropic").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

class TokenizerWrapper:
    def __init__(self, compute_tokens_func):
        self.compute_tokens_func = compute_tokens_func

    def encode(self, text: str) -> int:
        return self.compute_tokens_func([{"role": "user", "content": text}])

class ClaudeXChatEngine(Engine, AnthropicMixin):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        # In case we use EngineRepository.register to inject the api_key and model => dynamically change the engine at runtime
        if api_key is not None and model is not None:
            self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] = api_key
            self.config['NEUROSYMBOLIC_ENGINE_MODEL'] = model
        if self.id() != 'neurosymbolic':
            return # do not initialize if not neurosymbolic; avoids conflict with llama.cpp check in EngineRepository.register_from_package
        anthropic.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model = self.config['NEUROSYMBOLIC_ENGINE_MODEL']
        self.name = self.__class__.__name__
        self.tokenizer = TokenizerWrapper(self.compute_required_tokens)
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.client = anthropic.Anthropic(api_key=anthropic.api_key)

    def id(self) -> str:
        if self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
           self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('claude') and \
           ('3-7' not in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
            '4-0' not in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
            '4-1' not in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
            '4-5' not in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL')):
               return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            anthropic.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']

    def compute_required_tokens(self, messages) -> int:
        claude_messages, system_content = self._build_claude_messages(messages)

        if not claude_messages:
            return 0

        return self._count_claude_tokens(claude_messages, system_content)

    def _build_claude_messages(self, messages):
        claude_messages = []
        system_content = None

        for role, content_str in self._message_parts(messages):
            if role == 'system':
                system_content = content_str
                continue

            if role in ['user', 'assistant']:
                message_content = self._build_message_content(content_str)
                if message_content:
                    claude_messages.append(self._create_claude_message(role, message_content))

        return claude_messages, system_content

    def _message_parts(self, messages):
        for msg in messages:
            msg_parts = msg if isinstance(msg, list) else [msg]
            for part in msg_parts:
                yield self._extract_message_details(part)

    def _extract_message_details(self, part):
        if isinstance(part, str):
            return 'user', part

        if isinstance(part, dict):
            role = part.get('role')
            content_str = str(part.get('content', ''))
            return role, content_str

        msg = f"Unsupported message part type: {type(part)}"
        UserMessage(msg, raise_with=ValueError)
        raise ValueError(msg)

    def _build_message_content(self, content_str: str) -> list:
        message_content = []

        image_content = self._handle_image_content(content_str)
        message_content.extend(image_content)

        text_content = self._remove_vision_pattern(content_str)
        if text_content:
            message_content.append({
                "type": "text",
                "text": text_content
            })

        return message_content

    def _create_claude_message(self, role: str, message_content: list) -> dict:
        if len(message_content) == 1 and message_content[0].get('type') == 'text':
            return {
                'role': role,
                'content': message_content[0]['text']
            }

        return {
            'role': role,
            'content': message_content
        }

    def _count_claude_tokens(self, claude_messages: list, system_content: str | None) -> int:
        try:
            count_params = {
                'model': self.model,
                'messages': claude_messages
            }
            if system_content:
                count_params['system'] = system_content
            count_response = self.client.messages.count_tokens(**count_params)
            return count_response.input_tokens
        except Exception as e:
            UserMessage(f"Claude count_tokens failed: {e}")
            UserMessage(f"Error counting tokens for Claude: {e!s}", raise_with=RuntimeError)

    def compute_remaining_tokens(self, _prompts: list) -> int:
        UserMessage('Method not implemented.', raise_with=NotImplementedError)

    def _handle_image_content(self, content: str) -> list:
        """Handle image content by processing vision patterns and returning image file data."""
        def extract_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.findall(pattern, text)

        image_files = []
        if '<<vision:' in content:
            parts = extract_pattern(content)
            for p in parts:
                img_ = p.strip()
                max_frames_spacing = 50
                max_used_frames = 10
                buffer, ext = encode_media_frames(img_)
                if len(buffer) > 1:
                    step = len(buffer) // max_frames_spacing # max frames spacing
                    frames = []
                    indices = list(range(0, len(buffer), step))[:max_used_frames]
                    for i in indices:
                        frames.append({'data': buffer[i], 'media_type': f'image/{ext}', 'type': 'base64'})
                    image_files.extend(frames)
                elif len(buffer) == 1:
                    image_files.append({'data': buffer[0], 'media_type': f'image/{ext}', 'type': 'base64'})
                else:
                    UserMessage('No frames found for image!')
        return image_files

    def _remove_vision_pattern(self, text: str) -> str:
        """Remove vision patterns from text."""
        pattern = r'<<vision:(.*?):>>'
        return re.sub(pattern, '', text)

    def forward(self, argument):
        kwargs = argument.kwargs
        system, messages = argument.prop.prepared_input
        payload = self._prepare_request_payload(argument)
        except_remedy = kwargs.get('except_remedy')

        try:
            res = self.client.messages.create(
                system=system,
                messages=messages,
                **payload
            )
        except Exception as e:
            if anthropic.api_key is None or anthropic.api_key == '':
                msg = 'Anthropic API key is not set. Please set it in the config file or pass it as an argument to the command method.'
                UserMessage(msg)
                if self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] == '':
                    UserMessage(msg, raise_with=ValueError)
                anthropic.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']

            callback = self.client.messages.create
            kwargs['model'] = kwargs.get('model', self.model)

            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                UserMessage(f'Error during generation. Caused by: {e}', raise_with=ValueError)

        if payload['stream']:
            res = list(res) # Unpack the iterator to a list
        metadata = {'raw_output': res}
        response_data = self._collect_response(res)

        if response_data.get('function_call'):
            metadata['function_call'] = response_data['function_call']

        text_output = response_data.get('text', '')
        if argument.prop.response_format:
            # Anthropic returns JSON in markdown format
            text_output = text_output.replace('```json', '').replace('```', '')

        return [text_output], metadata

    def _prepare_raw_input(self, argument):
        if not argument.prop.processed_input:
            msg = 'Need to provide a prompt instruction to the engine if `raw_input` is enabled!'
            UserMessage(msg)
            raise ValueError(msg)
        system = NOT_GIVEN
        prompt = copy(argument.prop.processed_input)
        if not isinstance(prompt, list):
            if not isinstance(prompt, dict):
                prompt = {'role': 'user', 'content': str(prompt)}
            prompt = [prompt]
        if len(prompt) > 1:
            # assert there are not more than 1 system instruction
            assert len([p for p in prompt if p['role'] == 'system']) <= 1, 'Only one system instruction is allowed!'
            for p in prompt:
                if p['role'] == 'system':
                    system = p['content']
                    prompt.remove(p)
                    break
        return system, prompt

    def prepare(self, argument):
        #@NOTE: OpenAI compatibility at high level
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        non_verbose_output = """<META_INSTRUCTION/>\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n\n"""
        image_files = self._handle_image_content(str(argument.prop.processed_input))
        has_image = len(image_files) > 0

        system = self._build_system_prompt(argument, has_image, non_verbose_output)
        user_text, user_prompt, image_blocks = self._build_user_prompt(argument, image_files)
        system, user_prompt = self._apply_self_prompt_if_needed(argument, system, user_text, image_blocks, user_prompt)

        argument.prop.prepared_input = (system, [user_prompt])

    def _build_system_prompt(self, argument, has_image: bool, non_verbose_output: str) -> str:
        system = self._build_system_prefix(argument, non_verbose_output)
        system = self._append_context_sections(system, argument)
        system = self._append_instruction_section(system, argument, has_image)
        return self._append_template_suffix(system, argument)

    def _build_system_prefix(self, argument, non_verbose_output: str) -> str:
        system = ""
        if argument.prop.suppress_verbose_output:
            system += non_verbose_output

        system = f'{system}\n' if system and len(system) > 0 else ''

        if argument.prop.response_format:
            response_format = argument.prop.response_format
            assert response_format.get('type') is not None, 'Response format type is required! Expected format `{"type": str}`! The str value will be passed to the engine. Refer to the Anthropic documentation for more information: https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency#example-standardizing-customer-feedback'
            system += non_verbose_output
            system += f'<RESPONSE_FORMAT/>\n{response_format["type"]}\n\n'

        return system

    def _append_context_sections(self, system: str, argument) -> str:
        ref = argument.prop.instance
        static_context, dynamic_context = ref.global_context

        if len(static_context) > 0:
            system += f"<STATIC_CONTEXT/>\n{static_context}\n\n"

        if len(dynamic_context) > 0:
            system += f"<DYNAMIC_CONTEXT/>\n{dynamic_context}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system += f"<ADDITIONAL_CONTEXT/>\n{payload!s}\n\n"

        examples: list[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"<EXAMPLES/>\n{examples!s}\n\n"

        return system

    def _append_instruction_section(self, system: str, argument, has_image: bool) -> str:
        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            instruction_value = str(argument.prop.prompt)
            if has_image:
                instruction_value = self._remove_vision_pattern(instruction_value)
            system += f"<INSTRUCTION/>\n{instruction_value}\n\n"

        return system

    def _append_template_suffix(self, system: str, argument) -> str:
        if argument.prop.template_suffix:
            system += f' You will only generate content for the placeholder `{argument.prop.template_suffix!s}` following the instructions and the provided context information.\n\n'

        return system

    def _build_user_prompt(self, argument, image_files):
        suffix = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = self._remove_vision_pattern(suffix)

        user_text = f"{suffix}"
        if not user_text:
            user_text = "N/A"

        image_blocks = [{'type': 'image', 'source': image_file} for image_file in image_files]
        user_prompt = self._wrap_user_prompt_content(user_text, image_blocks)
        return user_text, user_prompt, image_blocks

    def _wrap_user_prompt_content(self, user_text: str, image_blocks: list[dict]) -> dict:
        if len(image_blocks) > 0:
            return {
                "role": "user",
                "content": [
                    *image_blocks,
                    {'type': 'text', 'text': user_text}
                ]
            }

        return {
            "role": "user",
            "content": user_text
        }

    def _apply_self_prompt_if_needed(self, argument, system: str, user_text: str, image_blocks: list[dict], user_prompt: dict):
        if not (argument.prop.instance._kwargs.get('self_prompt', False) or argument.prop.self_prompt):
            return system, user_prompt

        self_prompter = SelfPrompt()
        res = self_prompter({'user': user_text, 'system': system})
        if res is None:
            msg = "Self-prompting failed!"
            UserMessage(msg)
            raise ValueError(msg)

        updated_user_prompt = self._wrap_user_prompt_content(res['user'], image_blocks)
        return res['system'], updated_user_prompt

    def _prepare_request_payload(self, argument):
        kwargs = argument.kwargs
        model = kwargs.get('model', self.model)
        max_tokens = kwargs.get('max_tokens', self.max_response_tokens)
        stop = kwargs.get('stop', NOT_GIVEN)
        temperature = kwargs.get('temperature', 1)
        top_p = kwargs.get('top_p', NOT_GIVEN if temperature is not None else 1) #@NOTE:'You should either alter temperature or top_p, but not both.'
        top_k = kwargs.get('top_k', NOT_GIVEN)
        stream = kwargs.get('stream', True) # Do NOT remove this default value! Getting tons of API errors because they can't process requests >10m
        tools = kwargs.get('tools', NOT_GIVEN)
        tool_choice = kwargs.get('tool_choice', NOT_GIVEN)
        metadata_anthropic = kwargs.get('metadata', NOT_GIVEN)

        if stop != NOT_GIVEN and not isinstance(stop, list):
            stop = [stop]

        #@NOTE: Anthropic fails if stop is not raw string, so cast it to r'…'
        #       E.g. when we use defaults in core.py, i.e. stop=['\n']
        if stop != NOT_GIVEN:
            stop = [r'{s}' for s in stop]

        return {
            "model": model,
            "max_tokens": max_tokens,
            "stop_sequences": stop,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream,
            "metadata": metadata_anthropic,
            "tools": tools,
            "tool_choice": tool_choice
        }

    def _collect_response(self, res):
        if isinstance(res, list):
            return self._collect_streaming_response(res)

        if isinstance(res, Message):
            return self._collect_message_response(res)

        UserMessage(f"Unexpected response type from Anthropic API: {type(res)}", raise_with=ValueError)
        return {}

    def _collect_streaming_response(self, res):
        text_parts = []
        tool_calls_raw = []
        active_tool_calls = {}

        for chunk in res:
            if isinstance(chunk, RawContentBlockStartEvent):
                self._start_tool_call(chunk, active_tool_calls)
            elif isinstance(chunk, RawContentBlockDeltaEvent):
                self._update_stream_chunk(chunk, text_parts, active_tool_calls)
            elif isinstance(chunk, RawContentBlockStopEvent):
                tool_call = self._finish_tool_call(chunk, active_tool_calls)
                if tool_call is not None:
                    tool_calls_raw.append(tool_call)

        text_content = ''.join(text_parts)
        function_call_data = self._build_function_call_data(tool_calls_raw)

        return {
            "text": text_content,
            "function_call": function_call_data
        }

    def _start_tool_call(self, chunk, active_tool_calls: dict):
        if isinstance(chunk.content_block, ToolUseBlock):
            active_tool_calls[chunk.index] = {
                'id': chunk.content_block.id,
                'name': chunk.content_block.name,
                'input_json_str': ""
            }

    def _update_stream_chunk(self, chunk, text_parts: list, active_tool_calls: dict):
        if isinstance(chunk.delta, TextDelta):
            text_parts.append(chunk.delta.text)
        elif isinstance(chunk.delta, InputJSONDelta) and chunk.index in active_tool_calls:
            active_tool_calls[chunk.index]['input_json_str'] += chunk.delta.partial_json

    def _finish_tool_call(self, chunk, active_tool_calls: dict):
        if chunk.index not in active_tool_calls:
            return None

        tool_call_info = active_tool_calls.pop(chunk.index)
        try:
            tool_call_info['input'] = json.loads(tool_call_info['input_json_str'])
        except json.JSONDecodeError as e:
            UserMessage(f"Failed to parse JSON for tool call {tool_call_info['name']}: {e}. Raw JSON: '{tool_call_info['input_json_str']}'")
            tool_call_info['input'] = {}
        return tool_call_info

    def _build_function_call_data(self, tool_calls_raw: list | None) -> dict | None:
        if not tool_calls_raw:
            return None

        if len(tool_calls_raw) > 1:
            UserMessage("Multiple tool calls detected in the stream but only the first one will be processed.")

        tool_call = tool_calls_raw[0]
        return {
            'name': tool_call['name'],
            'arguments': tool_call['input']
        }

    def _collect_message_response(self, res: Message):
        text_parts = []
        function_call_data = None
        hit_tool_use = False

        for content_block in res.content:
            if isinstance(content_block, TextBlock):
                text_parts.append(content_block.text)
            elif isinstance(content_block, ToolUseBlock):
                if hit_tool_use:
                    UserMessage("Multiple tool use blocks detected in the response but only the first one will be processed.")
                else:
                    function_call_data = {
                        'name': content_block.name,
                        'arguments': content_block.input
                    }
                    hit_tool_use = True

        return {
            "text": ''.join(text_parts),
            "function_call": function_call_data
        }
