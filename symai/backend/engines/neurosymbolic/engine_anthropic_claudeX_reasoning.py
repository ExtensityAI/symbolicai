import json
import logging
import re
from copy import copy, deepcopy
from typing import List, Optional

import anthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import (InputJSONDelta, Message,
                             RawContentBlockDeltaEvent,
                             RawContentBlockStartEvent,
                             RawContentBlockStopEvent, TextBlock, TextDelta,
                             ThinkingBlock, ThinkingDelta, ToolUseBlock)

from ....components import SelfPrompt
from ....misc.console import ConsoleStyle
from ....symbol import Symbol
from ....utils import CustomUserWarning, encode_media_frames
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
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
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
            '4-0' in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL')):
               return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            anthropic.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']

    def compute_required_tokens(self, messages) -> int:
        claude_messages = []
        system_content = None

        for msg in messages:
            if not isinstance(msg, list):
                msg = [msg]
            for part in msg:
                if isinstance(part, str):
                    role = 'user'
                    content_str = part
                elif isinstance(part, dict):
                    role = part.get('role')
                    content_str = str(part.get('content', ''))
                else:
                    CustomUserWarning(f"Unsupported message part type: {type(part)}", raise_with=ValueError)

                if role == 'system':
                    system_content = content_str
                    continue

                if role in ['user', 'assistant']:
                    message_content = []

                    image_content = self._handle_image_content(content_str)
                    message_content.extend(image_content)

                    text_content = self._remove_vision_pattern(content_str)
                    if text_content:
                        message_content.append({
                            "type": "text",
                            "text": text_content
                        })

                    if message_content:
                        if len(message_content) == 1 and message_content[0].get('type') == 'text':
                            claude_messages.append({
                                'role': role,
                                'content': message_content[0]['text']
                            })
                        else:
                            claude_messages.append({
                                'role': role,
                                'content': message_content
                            })

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
            logging.error(f"Claude count_tokens failed: {e}")
            CustomUserWarning(f"Error counting tokens for Claude: {str(e)}", raise_with=RuntimeError)

    def compute_remaining_tokens(self, prompts: list) -> int:
        CustomUserWarning('Method not implemented.', raise_with=NotImplementedError)

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
                    CustomUserWarning(f'No frames found for image!')
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
                logging.error(msg)
                if self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] == '':
                    CustomUserWarning(msg, raise_with=ValueError)
                anthropic.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']

            callback = self.client.messages.create
            kwargs['model'] = kwargs['model'] if 'model' in kwargs else self.model

            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                CustomUserWarning(f"Anthropic API request failed: {e}", raise_with=ValueError)

        if payload['stream']:
            res = [_ for _ in res] # Unpack the iterator to a list
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
            raise ValueError('Need to provide a prompt instruction to the engine if `raw_input` is enabled!')
        system = NOT_GIVEN
        prompt = copy(argument.prop.processed_input)
        if type(prompt) != list:
            if type(prompt) != dict:
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
        user:   str = ""
        system: str = ""

        if argument.prop.suppress_verbose_output:
            system += _non_verbose_output
        system = f'{system}\n' if system and len(system) > 0 else ''

        if argument.prop.response_format:
            _rsp_fmt = argument.prop.response_format
            if not (_rsp_fmt.get('type') is not None):
                CustomUserWarning('Response format type is required! Expected format `{"type": "json_object"}` or other supported types. Refer to Anthropic documentation for details.', raise_with=AssertionError)
            system += _non_verbose_output
            system += f'<RESPONSE_FORMAT/>\n{_rsp_fmt["type"]}\n\n'

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"<STATIC_CONTEXT/>\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"<DYNAMIC_CONTEXT/>\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system += f"<ADDITIONAL_CONTEXT/>\n{str(payload)}\n\n"

        examples: List[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"<EXAMPLES/>\n{str(examples)}\n\n"

        image_files = self._handle_image_content(str(argument.prop.processed_input))

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            if len(image_files) > 0:
                val = self._remove_vision_pattern(val)
            system += f"<INSTRUCTION/>\n{val}\n\n"

        suffix: str = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = self._remove_vision_pattern(suffix)

        user += f"{suffix}"

        if not len(user):
            # Anthropic doesn't allow empty user prompts; force it
            user = "N/A"

        if argument.prop.template_suffix:
            system += f' You will only generate content for the placeholder `{str(argument.prop.template_suffix)}` following the instructions and the provided context information.\n\n'

        if len(image_files) > 0:
            images = [{ 'type': 'image', "source": im } for im in image_files]
            user_prompt = { "role": "user", "content": [
                *images,
                { 'type': 'text', 'text': user }
            ]}
        else:
            user_prompt = { "role": "user", "content": user }

        # First check if the `Symbol` instance has the flag set, otherwise check if it was passed as an argument to a method
        if argument.prop.instance._kwargs.get('self_prompt', False) or argument.prop.self_prompt:
            self_prompter = SelfPrompt()

            res = self_prompter(
                {'user': user, 'system': system},
                max_tokens=argument.kwargs.get('max_tokens', self.max_response_tokens),
                thinking=argument.kwargs.get('thinking', NOT_GIVEN),
            )
            if res is None:
                CustomUserWarning("Self-prompting failed to return a response.", raise_with=ValueError)

            if len(image_files) > 0:
                user_prompt = { "role": "user", "content": [
                    *images,
                    { 'type': 'text', 'text': res['user'] }
                ]}
            else:
                user_prompt = { "role": "user", "content": res['user'] }

            system = res['system']

        argument.prop.prepared_input = (system, [user_prompt])

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

        if stop != NOT_GIVEN and type(stop) != list:
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
            thinking_content = ''
            text_content = ''
            tool_calls_raw = []
            active_tool_calls = {}

            for chunk in res:
                if isinstance(chunk, RawContentBlockStartEvent):
                    if isinstance(chunk.content_block, ToolUseBlock):
                        active_tool_calls[chunk.index] = {
                            'id': chunk.content_block.id,
                            'name': chunk.content_block.name,
                            'input_json_str': ""
                        }
                elif isinstance(chunk, RawContentBlockDeltaEvent):
                    if isinstance(chunk.delta, ThinkingDelta):
                        thinking_content += chunk.delta.thinking
                    elif isinstance(chunk.delta, TextDelta):
                        text_content += chunk.delta.text
                    elif isinstance(chunk.delta, InputJSONDelta):
                        if chunk.index in active_tool_calls:
                            active_tool_calls[chunk.index]['input_json_str'] += chunk.delta.partial_json
                elif isinstance(chunk, RawContentBlockStopEvent):
                    if chunk.index in active_tool_calls:
                        tool_call_info = active_tool_calls.pop(chunk.index)
                        try:
                            tool_call_info['input'] = json.loads(tool_call_info['input_json_str'])
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse JSON for tool call {tool_call_info['name']}: {e}. Raw JSON: '{tool_call_info['input_json_str']}'")
                            tool_call_info['input'] = {}
                        tool_calls_raw.append(tool_call_info)

            function_call_data = None
            if tool_calls_raw:
                if len(tool_calls_raw) > 1:
                    CustomUserWarning("Multiple tool calls detected in the stream but only the first one will be processed.")
                function_call_data = {
                    'name': tool_calls_raw[0]['name'],
                    'arguments': tool_calls_raw[0]['input']
                }

            return {
                "thinking": thinking_content,
                "text": text_content,
                "function_call": function_call_data
            }

        # Non-streamed response (res is a Message object)
        if isinstance(res, Message):
            thinking_content = ''
            text_content = ''
            function_call_data = None
            hit = False

            for content_block in res.content:
                if isinstance(content_block, ThinkingBlock):
                    thinking_content += content_block.thinking
                elif isinstance(content_block, TextBlock):
                    text_content += content_block.text
                elif isinstance(content_block, ToolUseBlock):
                    if hit:
                        CustomUserWarning("Multiple tool use blocks detected in the response but only the first one will be processed.")
                    else:
                        function_call_data = {
                            'name': content_block.name,
                            'arguments': content_block.input
                        }
                        hit = True
            return {
                "thinking": thinking_content,
                "text": text_content,
                "function_call": function_call_data
            }

        CustomUserWarning(f"Unexpected response type from Anthropic API: {type(res)}", raise_with=ValueError)
