import logging
import re
from copy import deepcopy
from typing import List, Optional

import anthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import (Message, RawContentBlockDeltaEvent, TextBlock,
                             TextDelta, ThinkingBlock, ThinkingDelta)

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
        self.tokenizer = None # TODO: https://docs.anthropic.com/en/docs/build-with-claude/token-counting
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.client = anthropic.Anthropic(api_key=anthropic.api_key)

    def id(self) -> str:
        if self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
           self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('claude') and \
           '3-7' in self.config.get('NEUROSYMBOLIC_ENGINE_MODEL'):
               return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            anthropic.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']

    def compute_required_tokens(self, messages):
        # TODO: https://docs.anthropic.com/en/api/messages-count-tokens
        raise NotImplementedError('Method not implemented.')

    def compute_remaining_tokens(self, prompts: list) -> int:
        raise NotImplementedError('Method not implemented.')

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
                    raise Exception(msg) from e
                anthropic.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']

            callback = self.client.messages.create
            kwargs['model'] = kwargs['model'] if 'model' in kwargs else self.model

            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                raise e

        metadata = {'raw_output': res}
        output = self._collect_response(res, stream=payload['stream'])
        if len(output['thinking']) > 0:
            # Means reasoning was enabled
            metadata['thinking'] = output['thinking']

        if argument.prop.response_format:
            # Anthropic returns JSON in markdown format
            output['text'] = output['text'].replace('```json', '').replace('```', '')

        return [output['text']], metadata

    def prepare(self, argument):
        #@NOTE: OpenAI compatibility at high level
        if argument.prop.raw_input:
            if not argument.prop.processed_input:
                raise ValueError('Need to provide a prompt instruction to the engine if `raw_input` is enabled!')
            system = NOT_GIVEN
            prompt = argument.prop.processed_input
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
            argument.prop.prepared_input = system, prompt
            return

        _non_verbose_output = """<META_INSTRUCTION/>\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n\n"""
        user:   str = ""
        system: str = ""

        if argument.prop.suppress_verbose_output:
            system += _non_verbose_output
        system = f'{system}\n' if system and len(system) > 0 else ''

        if argument.prop.response_format:
            _rsp_fmt = argument.prop.response_format
            assert _rsp_fmt.get('type') is not None, 'Response format type is required! Expected format `{"type": str}`! The str value will be passed to the engine. Refer to the Anthropic documentation for more information: https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency#example-standardizing-customer-feedback'
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

        def extract_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.findall(pattern, text)

        def remove_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.sub(pattern, '', text)

        image_files = []
        # pre-process prompt if contains image url
        if '<<vision:' in str(argument.prop.processed_input):
            parts = extract_pattern(str(argument.prop.processed_input))
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

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            if len(image_files) > 0:
                val = remove_pattern(val)
            system += f"<INSTRUCTION/>\n{val}\n\n"

        suffix: str = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = remove_pattern(suffix)

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

            res = self_prompter({'user': user, 'system': system})
            if res is None:
                raise ValueError("Self-prompting failed!")

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
        thinking = kwargs.get('thinking', NOT_GIVEN)
        top_p = kwargs.get('top_p', NOT_GIVEN if temperature is not None else 1) #@NOTE:'You should either alter temperature or top_p, but not both.'
        top_k = kwargs.get('top_k', NOT_GIVEN)
        stream = kwargs.get('stream', NOT_GIVEN)
        tools = kwargs.get('tools', NOT_GIVEN)
        tool_choice = kwargs.get('tool_choice', NOT_GIVEN)
        metadata_anthropic = kwargs.get('metadata', NOT_GIVEN)

        if stop != NOT_GIVEN and type(stop) != list:
            stop = [stop]

        #@NOTE: Anthropic fails if stop is not raw string, so cast it to r'…'
        #       E.g. when we use defaults in core.py, i.e. stop=['\n']
        if stop != NOT_GIVEN:
            stop = [r'{s}' for s in stop]

        # set max_tokens based on thinking
        if thinking != NOT_GIVEN:
            max_tokens = kwargs.get('max_tokens', self.max_response_tokens[1])
        else:
            max_tokens = kwargs.get('max_tokens', self.max_response_tokens[0])

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

    def _collect_response(self, res, stream):
        if stream:
            thinking_content = ''
            text_content = ''
            for chunk in res:
                if isinstance(chunk, RawContentBlockDeltaEvent):
                    if isinstance(chunk.delta, ThinkingDelta):
                        thinking_content += chunk.delta.thinking
                        continue
                    if isinstance(chunk.delta, TextDelta):
                        text_content += chunk.delta.text
            return {
                "thinking": thinking_content,
                "text": text_content
            }

        if isinstance(res, Message):
            thinking_content = ''
            text_content = ''
            for content in res.content:
                if isinstance(content, ThinkingBlock):
                    thinking_content += content.thinking
                    continue
                if isinstance(content, TextBlock):
                    text_content += content.text
            return {
                "thinking": thinking_content,
                "text": text_content
            }

        raise ValueError("Unexpected response type from Anthropic API")
