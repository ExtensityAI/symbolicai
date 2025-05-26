import json
import logging
import re
from copy import deepcopy
from typing import List, Optional

import requests

from ....utils import CustomUserWarning
from ...base import Engine
from ...settings import SYMAI_CONFIG, SYMSERVER_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class HFTokenizer:
    _server_endpoint = f"http://{SYMSERVER_CONFIG.get('host')}:{SYMSERVER_CONFIG.get('port')}"

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> List[int]:
        res = requests.post(f"{HFTokenizer._server_endpoint}/tokenize", json={
            "input": text,
            "add_special_tokens": add_special_tokens,
        })

        if res.status_code != 200:
            CustomUserWarning(f"Request failed with status code: {res.status_code}", raise_with=ValueError)

        res = res.json()

        return res['tokens']

    @staticmethod
    def decode(tokens: List[int], skip_special_tokens: bool = True) -> str:
        res = requests.post(f"{HFTokenizer._server_endpoint}/detokenize", json={
            "tokens": tokens,
            "skip_special_tokens": skip_special_tokens,
        })

        if res.status_code != 200:
            CustomUserWarning(f"Request failed with status code: {res.status_code}", raise_with=ValueError)

        res = res.json()

        return res['text']


class HFEngine(Engine):
    def __init__(self, model: Optional[str] = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        # In case we use EngineRepository.register to inject the api_key and model => dynamically change the engine at runtime
        if model is not None:
            self.config['NEUROSYMBOLIC_ENGINE_MODEL'] = model
        if self.id() != 'neurosymbolic':
            return
        if not SYMSERVER_CONFIG.get('online'):
            CustomUserWarning('You are using the huggingface engine, but the server endpoint is not started. Please start the server with `symserver [--args]` or run `symserver --help` to see the available options for this engine.', raise_with=ValueError)
        self.server_endpoint = f"http://{SYMSERVER_CONFIG.get('host')}:{SYMSERVER_CONFIG.get('port')}"
        self.tokenizer = HFTokenizer # backwards compatibility with how we handle tokenization, i.e. self.tokenizer().encode(...)
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('huggingface'):
            return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']
        if 'seed' in kwargs:
            self.seed = kwargs['seed']
        if 'except_remedy' in kwargs:
            self.except_remedy = kwargs['except_remedy']

    def compute_required_tokens(self, messages) -> int:
        CustomUserWarning('Not implemented for HFEngine. Please use the tokenizer directly to compute tokens.', raise_with=NotImplementedError)
    def compute_remaining_tokens(self, prompts: list) -> int:
        CustomUserWarning('Not implemented for HFEngine. Please use the tokenizer directly to compute tokens.', raise_with=NotImplementedError)
    def forward(self, argument):
        kwargs  = argument.kwargs
        prompts = argument.prop.prepared_input

        stop               = kwargs.get('stop')
        seed               = kwargs.get('seed')
        temperature        = kwargs.get('temperature', 1.)
        top_p              = kwargs.get('top_p', 1.)
        top_k              = kwargs.get('top_k', 50)
        max_tokens         = kwargs.get('max_tokens', 2048)
        max_tokens_forcing = kwargs.get('max_tokens_forcing')
        logprobs           = kwargs.get('logprobs', True)
        do_sample          = kwargs.get('do_sample', True)
        num_beams          = kwargs.get('num_beams', 1)
        num_beam_groups    = kwargs.get('num_beam_groups', 1)
        eos_token_id       = kwargs.get('eos_token_id')
        except_remedy      = kwargs.get('except_remedy')

        try:
            res = requests.post(f"{self.server_endpoint}/chat", json={
                "messages": prompts,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
                "seed": seed,
                "max_tokens": max_tokens,
                "max_tokens_forcing": max_tokens_forcing,
                "top_k": top_k,
                "logprobs": logprobs,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "num_beam_groups": num_beam_groups,
                "eos_token_id": eos_token_id,
            })

            if res.status_code != 200:
                CustomUserWarning(f"Request failed with status code: {res.status_code}", raise_with=ValueError)

            res = res.json()

        except Exception as e:
            if except_remedy is not None:
                res = except_remedy(self, e, argument)
            else:
                CustomUserWarning(f"Request failed with exception: {str(e)}", raise_with=ValueError)

        metadata = {'raw_output': res}

        rsp    = [r['message']['content'] for r in res['choices']]
        output = rsp if isinstance(prompts, list) else rsp[0]
        return output, metadata

    def _prepare_raw_input(self, argument):
        if not argument.prop.processed_input:
            CustomUserWarning('Need to provide a prompt instruction to the engine if raw_input is enabled.', raise_with=ValueError)
        value = argument.prop.processed_input
        if type(value) != list:
            if type(value) != dict:
                value = {'role': 'user', 'content': str(value)}
            value = [value]
        return value

    def prepare(self, argument):
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        _non_verbose_output = """<META_INSTRUCTION/>\n You will NOT output verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. You will consider well formatted output, e.g. for sentences you will use punctuation, spaces, etc. or for code indentation, etc.\n"""

        #@TODO: Non-trivial how to handle user/system/assistant roles;
        user = ""

        if argument.prop.suppress_verbose_output:
            user += _non_verbose_output
        user = f'{user}\n' if user and len(user) > 0 else ''

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            user += f"<STATIC_CONTEXT/>\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            user += f"<DYNAMIC_CONTEXT/>\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            user += f"<ADDITIONAL_CONTEXT/>\n{str(payload)}\n\n"

        examples: List[str] = argument.prop.examples
        if examples and len(examples) > 0:
            user += f"<EXAMPLES/>\n{str(examples)}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            user += f"<INSTRUCTION/>\n{str(argument.prop.prompt)}\n\n"

        if argument.prop.template_suffix:
            user += f" You will only generate content for the placeholder `{str(argument.prop.template_suffix)}` following the instructions and the provided context information.\n\n"

        user += str(argument.prop.processed_input)

        argument.prop.prepared_input = [
            { "role": "user", "content": user },
        ]
