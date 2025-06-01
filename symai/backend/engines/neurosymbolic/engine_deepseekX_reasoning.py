import logging
import re
from copy import deepcopy
from typing import List, Optional

from annotated_types import Not
from openai import OpenAI

from ....components import SelfPrompt
from ....misc.console import ConsoleStyle
from ....symbol import Symbol
from ....utils import CustomUserWarning, encode_media_frames
from ...base import Engine
from ...mixin.deepseek import DeepSeekMixin
from ...settings import SYMAI_CONFIG

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class DeepSeekXReasoningEngine(Engine, DeepSeekMixin):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        # In case we use EngineRepository.register to inject the api_key and model => dynamically change the engine at runtime
        if api_key is not None and model is not None:
            self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] = api_key
            self.config['NEUROSYMBOLIC_ENGINE_MODEL'] = model
        if self.id() != 'neurosymbolic':
            return # do not initialize if not neurosymbolic; avoids conflict with llama.cpp check in EngineRepository.register_from_package
        self.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model = self.config['NEUROSYMBOLIC_ENGINE_MODEL']
        self.name = self.__class__.__name__
        self.tokenizer = None
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.seed = None

        try:
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        except Exception as e:
            CustomUserWarning(f'Failed to initialize the DeepSeek client. Please check your library version. Caused by: {e}', raise_with=RuntimeError)

    def id(self) -> str:
        if self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
           self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('deepseek'):
            return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']
        if 'seed' in kwargs:
            self.seed = kwargs['seed']

    def compute_required_tokens(self, messages):
        CustomUserWarning('Method "compute_required_tokens" not implemented for DeepSeekXReasoningEngine.', raise_with=NotImplementedError)

    def compute_remaining_tokens(self, prompts: list) -> int:
        CustomUserWarning('Method "compute_remaining_tokens" not implemented for DeepSeekXReasoningEngine.', raise_with=NotImplementedError)

    def truncate(self, prompts: list[dict], truncation_percentage: float | None, truncation_type: str) -> list[dict]:
        CustomUserWarning('Method "truncate" not implemented for DeepSeekXReasoningEngine.', raise_with=NotImplementedError)

    def forward(self, argument):
        kwargs = argument.kwargs
        messages = argument.prop.prepared_input
        payload = self._prepare_request_payload(argument)
        except_remedy = kwargs.get('except_remedy')

        try:
            res = self.client.chat.completions.create(messages=messages, **payload)

        except Exception as e:
            if self.api_key is None or self.api_key == '':
                msg = 'DeepSeek API key is not set. Please set it in the config file or pass it as an argument to the command method.'
                logging.error(msg)
                if self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] == '':
                    CustomUserWarning(msg, raise_with=ValueError)
                self.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']

            callback = self.client.chat.completions.create
            kwargs['model'] = kwargs['model'] if 'model' in kwargs else self.model

            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                CustomUserWarning(f"DeepSeek API request failed. Original error: {e}", raise_with=ValueError)

        reasoning_content = res.choices[0].message.reasoning_content
        content = res.choices[0].message.content
        metadata = {'raw_output': res, 'thinking': reasoning_content}

        return [content], metadata

    def _prepare_raw_input(self, argument):
        if not argument.prop.processed_input:
            CustomUserWarning('A prompt instruction is required for DeepSeekXReasoningEngine when raw_input is enabled.', raise_with=ValueError)
        value = argument.prop.processed_input
        # convert to dict if not already
        if type(value) != list:
            if type(value) != dict:
                value = {'role': 'user', 'content': str(value)}
            value = [value]
        return value

    def prepare(self, argument):
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
                CustomUserWarning('Response format type is required! Expected format `{"type": "json_object"}` or other supported types.', raise_with=AssertionError)
            system += _non_verbose_output
            system += f'<RESPONSE_FORMAT/>\n{_rsp_fmt["type"]}\n\n'

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"<STATIC CONTEXT/>\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"<DYNAMIC CONTEXT/>\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system += f"<ADDITIONAL CONTEXT/>\n{str(payload)}\n\n"

        examples: List[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"<EXAMPLES/>\n{str(examples)}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            system += f"<INSTRUCTION/>\n{val}\n\n"

        user += f"{str(argument.prop.processed_input)}"

        if argument.prop.template_suffix:
            system += f' You will only generate content for the placeholder `{str(argument.prop.template_suffix)}` following the instructions and the provided context information.\n\n'

        user_prompt = { "role": "user", "content": user }

        # First check if the `Symbol` instance has the flag set, otherwise check if it was passed as an argument to a method
        if argument.prop.instance._kwargs.get('self_prompt', False) or argument.prop.self_prompt:
            self_prompter = SelfPrompt()

            res = self_prompter({'user': user, 'system': system})
            if res is None:
                CustomUserWarning("Self-prompting failed for DeepSeekXReasoningEngine.", raise_with=ValueError)

            user_prompt = { "role": "user", "content": res['user'] }
            system = res['system']

        argument.prop.prepared_input = [
            { "role": "system", "content": system },
            user_prompt,
        ]

    def _prepare_request_payload(self, argument):
        """Prepares the request payload from the argument."""
        kwargs = argument.kwargs
        # 16/03/2025
        # Not Supported Features：Function Call、Json Output、FIM (Beta)
        # Not Supported Parameters：temperature、top_p、presence_penalty、frequency_penalty、logprobs、top_logprobs
        return {
            "model": kwargs.get('model', self.model),
            "seed": kwargs.get('seed', self.seed),
            "max_tokens": kwargs.get('max_tokens', self.max_response_tokens),
            "stop": kwargs.get('stop', '<|endoftext|>'),
            "n": kwargs.get('n', 1),
            "logit_bias": kwargs.get('logit_bias'),
        }
