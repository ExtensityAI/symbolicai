import json
import logging
import re
from copy import deepcopy

import openai

from ....components import SelfPrompt
from ....core_ext import retry
from ....utils import CustomUserWarning
from ...base import Engine
from ...settings import SYMAI_CONFIG

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class GroqEngine(Engine):
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
        self.model = self.config['NEUROSYMBOLIC_ENGINE_MODEL'].replace('groq:', '')
        self.seed = None
        self.name = self.__class__.__name__

        try:
            self.client = openai.OpenAI(api_key=openai.api_key, base_url="https://api.groq.com/openai/v1")
        except Exception as e:
            CustomUserWarning(f'Failed to initialize OpenAI client. Please check your OpenAI library version. Caused by: {e}', raise_with=ValueError)

    def id(self) -> str:
        if self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
           self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('groq'):
                return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            openai.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL'].replace('groq:', '')
        if 'seed' in kwargs:
            self.seed = kwargs['seed']

    def compute_required_tokens(self, messages):
        raise NotImplementedError("Token counting not implemented for this engine.")

    def compute_remaining_tokens(self, prompts: list) -> int:
        raise NotImplementedError("Token counting not implemented for this engine.")

    def _handle_image_content(self, content: str) -> list:
        """Handle image content by processing vision patterns and returning image file data."""
        def extract_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.findall(pattern, text)
        raise NotImplementedError("Image content handling not implemented for this engine.")


    def _remove_vision_pattern(self, text: str) -> str:
        """Remove vision patterns from text."""
        pattern = r'<<vision:(.*?):>>'
        return re.sub(pattern, '', text)

    def _extract_thinking_content(self, output: list[str]) -> tuple[str | None, list[str]]:
        """Extract thinking content from model output if present and return cleaned output."""
        if not output or len(output) == 0:
            return None, output

        content = output[0]
        if not content:
            return None, output

        think_pattern = r'<think>(.*?)</think>'
        match = re.search(think_pattern, content, re.DOTALL)

        thinking_content = None
        if match:
            thinking_content = match.group(1).strip()
            if not thinking_content:
                thinking_content = None

        cleaned_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
        cleaned_output = [cleaned_content] + output[1:]

        return thinking_content, cleaned_output

    # cumulative wait time is ~113s
    @retry(tries=8, delay=0.5, backoff=2, max_delay=30, jitter=(0, 0.5))
    def forward(self, argument):
        kwargs = argument.kwargs
        messages = argument.prop.prepared_input
        payload = self._prepare_request_payload(messages, argument)
        except_remedy = kwargs.get('except_remedy')

        try:
            res = self.client.chat.completions.create(**payload)

        except Exception as e:
            if openai.api_key is None or openai.api_key == '':
                msg = 'Groq API key is not set. Please set it in the config file or pass it as an argument to the command method.'
                logging.error(msg)
                if self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] == '':
                    CustomUserWarning(msg, raise_with=ValueError)
                openai.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']

            callback = self.client.chat.completions.create
            kwargs['model'] = kwargs['model'] if 'model' in kwargs else self.model

            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                CustomUserWarning(f'Error during generation. Caused by: {e}', raise_with=ValueError)

        metadata = {'raw_output': res}
        if payload.get('tools'):
            metadata = self._process_function_calls(res, metadata)

        output = [r.message.content for r in res.choices]
        thinking, output = self._extract_thinking_content(output)
        if thinking:
            metadata['thinking'] = thinking

        return output, metadata

    def _prepare_raw_input(self, argument):
        if not argument.prop.processed_input:
            CustomUserWarning('Need to provide a prompt instruction to the engine if raw_input is enabled.', raise_with=ValueError)
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
            assert _rsp_fmt.get('type') is not None, 'Expected format `{ "type": "json_object" }`! We are using the OpenAI compatible API for Groq. See more here: https://console.groq.com/docs/tool-use'
            if _rsp_fmt["type"] == "json_object":
                # OpenAI docs:
                    # "Important: when using JSON mode, you must also instruct the model
                    #  to produce JSON yourself via a system or user message"
                    #  Assuming this stays true even for this engine
                system += f'<RESPONSE_FORMAT/>\nYou are a helpful assistant designed to output JSON.\n\n'

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"<STATIC CONTEXT/>\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"<DYNAMIC CONTEXT/>\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system += f"<ADDITIONAL CONTEXT/>\n{str(payload)}\n\n"

        examples = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"<EXAMPLES/>\n{str(examples)}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            system += f"<INSTRUCTION/>\n{val}\n\n"

        suffix: str = str(argument.prop.processed_input)
        user += f"{suffix}"

        if argument.prop.template_suffix:
            system += f' You will only generate content for the placeholder `{str(argument.prop.template_suffix)}` following the instructions and the provided context information.\n\n'

        user_prompt = { "role": "user", "content": user }

        # First check if the `Symbol` instance has the flag set, otherwise check if it was passed as an argument to a method
        if argument.prop.instance._kwargs.get('self_prompt', False) or argument.prop.self_prompt:
            self_prompter = SelfPrompt()

            res = self_prompter({'user': user, 'system': system})
            if res is None:
                CustomUserWarning("Self-prompting failed!", raise_with=ValueError)

            user_prompt = { "role": "user", "content": res['user'] }
            system = res['system']

        argument.prop.prepared_input = [
            { "role": "system", "content": system },
            user_prompt,
        ]

    def _process_function_calls(self, res, metadata):
        hit = False
        if hasattr(res, 'choices') and res.choices:
            choice = res.choices[0]
            if hasattr(choice, 'message') and choice.message:
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        if hasattr(tool_call, 'function') and tool_call.function:
                            if hit:
                                CustomUserWarning("Multiple function calls detected in the response but only the first one will be processed.")
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

        for param in [
            "logprobs",
            "logit_bias",
            "top_logprobs",
            "search_settings",
            "stream",
            "stream_options",
        ]:
            if param in kwargs:
                CustomUserWarning(f"The parameter {param} is not supported by the Groq API. It will be ignored.")
                del kwargs[param]

        n = kwargs.get('n', 1)
        if n > 1:
            CustomUserWarning("If N is supplied, it must be equal to 1. We default to 1 to not crash your program.")
            n = 1

        payload = {
            "messages": messages,
            "model": kwargs.get('model', self.model),
            "seed": kwargs.get('seed', self.seed),
            "max_completion_tokens": kwargs.get('max_completion_tokens'),
            "stop": kwargs.get('stop'),
            "temperature": kwargs.get('temperature', 0.6), # Default temperature for Kimi K2 (https://huggingface.co/moonshotai/Kimi-K2-Instruct)
            "frequency_penalty": kwargs.get('frequency_penalty', 0),
            "presence_penalty": kwargs.get('presence_penalty', 0),
            "reasoning_effort": kwargs.get('reasoning_effort'), # Field available only for qwen3 models
            "service_tier": kwargs.get('service_tier', 'on_demand'),
            "top_p": kwargs.get('top_p', 1),
            "n": n,
            "tools": kwargs.get('tools'),
            "tool_choice": kwargs.get('tool_choice', 'none'),
            "response_format": kwargs.get('response_format'),
        }

        if not self.model.startswith('qwen'):
            del payload['reasoning_effort']

        return payload
