import asyncio
import json
import logging
from copy import deepcopy

import aiohttp
import httpx
import nest_asyncio
import requests

from ....core import Argument
from ....core_ext import retry
from ....utils import CustomUserWarning
from ...base import Engine
from ...settings import SYMAI_CONFIG, SYMSERVER_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class LlamaCppTokenizer:
    _server_endpoint = f"http://{SYMSERVER_CONFIG.get('--host')}:{SYMSERVER_CONFIG.get('--port')}"

    @staticmethod
    async def _encode(text: str) -> list[int]:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{LlamaCppTokenizer._server_endpoint}/extras/tokenize", json={
                "input": text,
            }) as res:
                if res.status != 200:
                    CustomUserWarning(f"Request failed with status code: {res.status}", raise_with=ValueError)
                res = await res.json()
                return res['tokens']

    @staticmethod
    def encode(text: str) -> list[int]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(LlamaCppTokenizer._encode(text))

    @staticmethod
    async def _decode(tokens: list[int]) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{LlamaCppTokenizer._server_endpoint}/extras/detokenize", json={
                "tokens": tokens,
            }) as res:
                if res.status != 200:
                    CustomUserWarning(f"Request failed with status code: {res.status}", raise_with=ValueError)
                res = await res.json()
                return res['text']

    @staticmethod
    def decode(tokens: list[int]) -> str:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(LlamaCppTokenizer._decode(tokens))


class LlamaCppEngine(Engine):
    _retry_params = {
        'tries': 5,
        'delay': 2,
        'max_delay': 60,
        'backoff': 2,
        'jitter': (1, 5),
        'graceful': True
    }
    _timeout_params = {
        'read': None,
        'connect': None,
    }
    def __init__(
            self,
            model: str | None = None,
            retry_params: dict = _retry_params,
            timeout_params: dict = _timeout_params,
        ):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        # In case we use EngineRepository.register to inject the api_key and model => dynamically change the engine at runtime
        if model is not None:
            self.config['NEUROSYMBOLIC_ENGINE_MODEL'] = model
        if self.id() != 'neurosymbolic':
            return
        if not SYMSERVER_CONFIG.get('online'):
            CustomUserWarning('You are using the llama.cpp engine, but the server endpoint is not started. Please start the server with `symserver [--args]` or run `symserver --help` to see the available options for this engine.', raise_with=ValueError)
        self.server_endpoint = f"http://{SYMSERVER_CONFIG.get('--host')}:{SYMSERVER_CONFIG.get('--port')}"
        self.tokenizer = LlamaCppTokenizer # backwards compatibility with how we handle tokenization, i.e. self.tokenizer().encode(...)
        self.timeout_params = self._validate_timeout_params(timeout_params)
        self.retry_params = self._validate_retry_params(retry_params)
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('llama'):
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
        #@TODO: quite non-trivial how to handle this with the llama.cpp server
        CustomUserWarning('Not implemented for llama.cpp!', raise_with=NotImplementedError)

    def compute_remaining_tokens(self, prompts: list) -> int:
        #@TODO: quite non-trivial how to handle this with the llama.cpp server
        CustomUserWarning('Not implemented for llama.cpp!', raise_with=NotImplementedError)

    def _validate_timeout_params(self, timeout_params):
        if not isinstance(timeout_params, dict):
            CustomUserWarning("timeout_params must be a dictionary", raise_with=ValueError)
        assert all(key in timeout_params for key in ['read', 'connect']), "Available keys: ['read', 'connect']"
        return timeout_params

    def _validate_retry_params(self, retry_params):
        if not isinstance(retry_params, dict):
            CustomUserWarning("retry_params must be a dictionary", raise_with=ValueError)
        assert all(key in retry_params for key in ['tries', 'delay', 'max_delay', 'backoff', 'jitter', 'graceful']), "Available keys: ['tries', 'delay', 'max_delay', 'backoff', 'jitter', 'graceful']"
        return retry_params

    @staticmethod
    def _get_event_loop() -> asyncio.AbstractEventLoop:
        """Gets or creates an event loop."""
        try:
            current_loop = asyncio.get_event_loop()
            if current_loop.is_closed():
                CustomUserWarning("Event loop is closed.", raise_with=RuntimeError)
            return current_loop
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop

    def _prepare_request_payload(self, argument: Argument) -> dict:
        """Prepares the request payload from the argument."""
        kwargs = argument.kwargs
        return {
            "messages": argument.prop.prepared_input,
            "temperature": kwargs.get('temperature', 0.6),
            "frequency_penalty": kwargs.get('frequency_penalty', 0),
            "presence_penalty": kwargs.get('presence_penalty', 0),
            "top_p": kwargs.get('top_p', 0.95),
            "min_p": kwargs.get('min_p', 0.05),
            "stop": kwargs.get('stop'),
            "seed": kwargs.get('seed'),
            "max_tokens": kwargs.get('max_tokens'),
            "top_k": kwargs.get('top_k', 40),
            "repeat_penalty": kwargs.get('repeat_penalty', 1),
            "logits_bias": kwargs.get('logits_bias'),
            "logprobs": kwargs.get('logprobs', False),
            "functions": kwargs.get('functions'),
            "function_call": kwargs.get('function_call'),
            "grammar": kwargs.get('grammar'),
            "response_format": kwargs.get('response_format'),
        }

    async def _arequest(self, payload: dict) -> dict:
        """Makes an async HTTP request to the llama.cpp server."""
        @retry(**self.retry_params)
        async def _make_request():
            timeout = aiohttp.ClientTimeout(
                sock_connect=self.timeout_params['connect'],
                sock_read=self.timeout_params['read']
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.server_endpoint}/v1/chat/completions",
                    json=payload
                ) as res:
                    if res.status != 200:
                        CustomUserWarning(f"Request failed with status code: {res.status}", raise_with=ValueError)
                    return await res.json()

        return await _make_request()

    def forward(self, argument):
        payload = self._prepare_request_payload(argument)

        nest_asyncio.apply()
        loop = self._get_event_loop()

        try:
            res = loop.run_until_complete(self._arequest(payload))
        except Exception as e:
            CustomUserWarning(f"Request failed with error: {str(e)}", raise_with=ValueError)

        metadata = {'raw_output': res}

        output = [r['message']['content'] for r in res['choices']]
        output = output if isinstance(argument.prop.prepared_input, list) else output[0]

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
        #       For instance Mixtral-8x7B can't use the system role with llama.cpp while other models can, or Mixtral-8x22B expects the conversation roles must
        #       alternate user/assistant/user/assistant/..., so how to handle this?
        #       For now just use the user, as one can rephrase the system from the user perspective.
        user:   str = ""

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

        examples: list[str] = argument.prop.examples
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
