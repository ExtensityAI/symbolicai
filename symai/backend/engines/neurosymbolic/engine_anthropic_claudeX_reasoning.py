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
    ThinkingBlock,
    ThinkingDelta,
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

class ClaudeXReasoningEngine(Engine, AnthropicMixin):
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
           ('3-7' in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') or \
            '4-0' in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') or \
            '4-1' in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') or \
            '4-5' in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL')):
               return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            anthropic.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']

    def compute_required_tokens(self, messages) -> int:
        claude_messages, system_content = self._normalize_messages_for_claude(messages)

        if not claude_messages:
            return 0

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

    def _normalize_messages_for_claude(self, messages):
        claude_messages = []
        system_content = None

        for msg in messages:
            msg_parts = msg if isinstance(msg, list) else [msg]
            for part in msg_parts:
                role, content_str = self._extract_role_and_content(part)
                if role == 'system':
                    system_content = content_str
                    continue

                if role in ['user', 'assistant']:
                    message_payload = self._build_message_payload(role, content_str)
                    if message_payload:
                        claude_messages.append(message_payload)

        return claude_messages, system_content

    def _extract_role_and_content(self, part):
        if isinstance(part, str):
            return 'user', part
        if isinstance(part, dict):
            return part.get('role'), str(part.get('content', ''))
        UserMessage(f"Unsupported message part type: {type(part)}", raise_with=ValueError)
        return None, ''

    def _build_message_payload(self, role, content_str):
        message_content = []

        image_content = self._handle_image_content(content_str)
        message_content.extend(image_content)

        text_content = self._remove_vision_pattern(content_str)
        if text_content:
            message_content.append({
                "type": "text",
                "text": text_content
            })

        if not message_content:
            return None

        if len(message_content) == 1 and message_content[0].get('type') == 'text':
            return {
                'role': role,
                'content': message_content[0]['text']
            }

        return {
            'role': role,
            'content': message_content
        }

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

        if response_data.get('thinking') and len(response_data['thinking']) > 0:
            metadata['thinking'] = response_data['thinking']

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

        _non_verbose_output = """<META_INSTRUCTION/>\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n\n"""
        image_files = self._handle_image_content(str(argument.prop.processed_input))
        system = self._build_system_prompt(argument, _non_verbose_output, image_files)
        user_text = self._build_user_text(argument, image_files)

        if not user_text:
            # Anthropic doesn't allow empty user prompts; force it
            user_text = "N/A"

        system, user_prompt = self._apply_self_prompt_if_needed(
            argument,
            system,
            user_text,
            image_files
        )

        argument.prop.prepared_input = (system, [user_prompt])

    def _build_system_prompt(self, argument, non_verbose_output, image_files):
        system = ""

        if argument.prop.suppress_verbose_output:
            system = f"{non_verbose_output}\n"

        if argument.prop.response_format:
            response_format = argument.prop.response_format
            if not (response_format.get('type') is not None):
                UserMessage('Response format type is required! Expected format `{"type": "json_object"}` or other supported types. Refer to Anthropic documentation for details.', raise_with=AssertionError)
            system += non_verbose_output
            system += f"<RESPONSE_FORMAT/>\n{response_format['type']}\n\n"

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"<STATIC_CONTEXT/>\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"<DYNAMIC_CONTEXT/>\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if payload:
            system += f"<ADDITIONAL_CONTEXT/>\n{payload!s}\n\n"

        examples: list[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"<EXAMPLES/>\n{examples!s}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            value = str(argument.prop.prompt)
            if len(image_files) > 0:
                value = self._remove_vision_pattern(value)
            system += f"<INSTRUCTION/>\n{value}\n\n"

        return self._append_template_suffix(system, argument.prop.template_suffix)

    def _build_user_text(self, argument, image_files):
        suffix = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = self._remove_vision_pattern(suffix)
        return suffix

    def _append_template_suffix(self, system, template_suffix):
        if template_suffix:
            return system + (
                f' You will only generate content for the placeholder `{template_suffix!s}` '
                'following the instructions and the provided context information.\n\n'
            )
        return system

    def _apply_self_prompt_if_needed(self, argument, system, user_text, image_files):
        if not self._is_self_prompt_enabled(argument):
            return system, self._format_user_prompt(user_text, image_files)

        self_prompter = SelfPrompt()
        response = self_prompter(
            {'user': user_text, 'system': system},
            max_tokens=argument.kwargs.get('max_tokens', self.max_response_tokens),
            thinking=argument.kwargs.get('thinking', NOT_GIVEN),
        )
        if response is None:
            UserMessage("Self-prompting failed to return a response.", raise_with=ValueError)

        updated_prompt = self._format_user_prompt(response['user'], image_files)
        return response['system'], updated_prompt

    def _is_self_prompt_enabled(self, argument):
        return argument.prop.instance._kwargs.get('self_prompt', False) or argument.prop.self_prompt

    def _format_user_prompt(self, user_text, image_files):
        if len(image_files) > 0:
            images = [{'type': 'image', 'source': im} for im in image_files]
            return {
                'role': 'user',
                'content': [
                    *images,
                    {'type': 'text', 'text': user_text}
                ]
            }

        return {'role': 'user', 'content': user_text}

    def _prepare_request_payload(self, argument):
        kwargs = argument.kwargs
        model = kwargs.get('model', self.model)
        stop = kwargs.get('stop', NOT_GIVEN)
        temperature = kwargs.get('temperature', 1)
        thinking_arg = kwargs.get('thinking', NOT_GIVEN)
        thinking = NOT_GIVEN
        if thinking_arg and isinstance(thinking_arg, dict):
            thinking = {
                "type": "enabled",
                "budget_tokens": thinking_arg.get("budget_tokens", 1024)
            }
        top_p = kwargs.get('top_p', NOT_GIVEN if temperature is not None else 1) #@NOTE:'You should either alter temperature or top_p, but not both.'
        top_k = kwargs.get('top_k', NOT_GIVEN)
        stream = kwargs.get('stream', True) # Do NOT remove this default value! Getting tons of API errors because they can't process requests >10m
        tools = kwargs.get('tools', NOT_GIVEN)
        tool_choice = kwargs.get('tool_choice', NOT_GIVEN)
        metadata_anthropic = kwargs.get('metadata', NOT_GIVEN)
        max_tokens = kwargs.get('max_tokens', self.max_response_tokens)

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
            "thinking": thinking,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream,
            "metadata": metadata_anthropic,
            "tools": tools,
            "tool_choice": tool_choice
        }

    def _collect_response(self, res):
        if isinstance(res, list):
            return self._collect_stream_response(res)

        if isinstance(res, Message):
            return self._collect_message_response(res)

        UserMessage(f"Unexpected response type from Anthropic API: {type(res)}", raise_with=ValueError)
        return {}

    def _collect_stream_response(self, response_chunks):
        accumulators = {'thinking': '', 'text': ''}
        tool_calls_raw = []
        active_tool_calls = {}

        for chunk in response_chunks:
            self._process_stream_chunk(chunk, accumulators, active_tool_calls, tool_calls_raw)

        function_call_data = self._extract_function_call(tool_calls_raw)
        return {
            'thinking': accumulators['thinking'],
            'text': accumulators['text'],
            'function_call': function_call_data
        }

    def _process_stream_chunk(self, chunk, accumulators, active_tool_calls, tool_calls_raw):
        if isinstance(chunk, RawContentBlockStartEvent):
            self._register_tool_call(chunk, active_tool_calls)
        elif isinstance(chunk, RawContentBlockDeltaEvent):
            self._handle_delta_chunk(chunk, accumulators, active_tool_calls)
        elif isinstance(chunk, RawContentBlockStopEvent):
            self._finalize_tool_call(chunk, active_tool_calls, tool_calls_raw)

    def _register_tool_call(self, chunk, active_tool_calls):
        if isinstance(chunk.content_block, ToolUseBlock):
            active_tool_calls[chunk.index] = {
                'id': chunk.content_block.id,
                'name': chunk.content_block.name,
                'input_json_str': ""
            }

    def _handle_delta_chunk(self, chunk, accumulators, active_tool_calls):
        if isinstance(chunk.delta, ThinkingDelta):
            accumulators['thinking'] += chunk.delta.thinking
        elif isinstance(chunk.delta, TextDelta):
            accumulators['text'] += chunk.delta.text
        elif isinstance(chunk.delta, InputJSONDelta) and chunk.index in active_tool_calls:
            active_tool_calls[chunk.index]['input_json_str'] += chunk.delta.partial_json

    def _finalize_tool_call(self, chunk, active_tool_calls, tool_calls_raw):
        if chunk.index not in active_tool_calls:
            return

        tool_call_info = active_tool_calls.pop(chunk.index)
        try:
            tool_call_info['input'] = json.loads(tool_call_info['input_json_str'])
        except json.JSONDecodeError as error:
            UserMessage(
                f"Failed to parse JSON for tool call {tool_call_info['name']}: {error}. Raw JSON: '{tool_call_info['input_json_str']}'"
            )
            tool_call_info['input'] = {}
        tool_calls_raw.append(tool_call_info)

    def _extract_function_call(self, tool_calls_raw):
        if not tool_calls_raw:
            return None

        if len(tool_calls_raw) > 1:
            UserMessage("Multiple tool calls detected in the stream but only the first one will be processed.")

        first_call = tool_calls_raw[0]
        return {
            'name': first_call['name'],
            'arguments': first_call['input']
        }

    def _collect_message_response(self, message):
        accumulators = {'thinking': '', 'text': ''}
        function_call_data = None
        tool_call_detected = False

        for content_block in message.content:
            function_call_data, tool_call_detected = self._process_message_block(
                content_block,
                accumulators,
                function_call_data,
                tool_call_detected
            )

        return {
            'thinking': accumulators['thinking'],
            'text': accumulators['text'],
            'function_call': function_call_data
        }

    def _process_message_block(self, content_block, accumulators, function_call_data, tool_call_detected):
        if isinstance(content_block, ThinkingBlock):
            accumulators['thinking'] += content_block.thinking
            return function_call_data, tool_call_detected

        if isinstance(content_block, TextBlock):
            accumulators['text'] += content_block.text
            return function_call_data, tool_call_detected

        if isinstance(content_block, ToolUseBlock):
            if tool_call_detected:
                UserMessage("Multiple tool use blocks detected in the response but only the first one will be processed.")
                return function_call_data, tool_call_detected

            return {
                'name': content_block.name,
                'arguments': content_block.input
            }, True

        return function_call_data, tool_call_detected
