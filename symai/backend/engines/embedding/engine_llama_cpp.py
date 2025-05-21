import asyncio
import logging
from multiprocessing import Value
from typing import Optional

import aiohttp
import nest_asyncio
import numpy as np

from ....core_ext import retry
from ....utils import CustomUserWarning
from ...base import Engine
from ...settings import SYMAI_CONFIG, SYMSERVER_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

class LlamaCppEmbeddingEngine(Engine):
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
            retry_params: dict = _retry_params,
            timeout_params: dict = _timeout_params
        ):
        super().__init__()
        self.config = SYMAI_CONFIG
        if self.id() != 'embedding':
            return
        if not SYMSERVER_CONFIG.get('online'):
            CustomUserWarning('You are using the llama.cpp embedding engine, but the server endpoint is not started. Please start the server with `symserver [--args]`.', raise_with=ValueError)

        self.server_endpoint = f"http://{SYMSERVER_CONFIG.get('--host')}:{SYMSERVER_CONFIG.get('--port')}"
        self.timeout_params = self._validate_timeout_params(timeout_params)
        self.retry_params = self._validate_retry_params(retry_params)
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config.get('EMBEDDING_ENGINE_MODEL') and self.config.get('EMBEDDING_ENGINE_MODEL').startswith('llama'):
            return 'embedding'
        return super().id()  # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'EMBEDDING_ENGINE_MODEL' in kwargs:
            self.model = kwargs['EMBEDDING_ENGINE_MODEL']

    def _validate_timeout_params(self, timeout_params):
        if not isinstance(timeout_params, dict):
            raise ValueError("timeout_params must be a dictionary")
        assert all(key in timeout_params for key in ['read', 'connect']), "Available keys: ['read', 'connect']"
        return timeout_params

    def _validate_retry_params(self, retry_params):
        if not isinstance(retry_params, dict):
            raise ValueError("retry_params must be a dictionary")
        assert all(key in retry_params for key in ['tries', 'delay', 'max_delay', 'backoff', 'jitter', 'graceful']), \
            "Available keys: ['tries', 'delay', 'max_delay', 'backoff', 'jitter', 'graceful']"
        return retry_params

    @staticmethod
    def _get_event_loop() -> asyncio.AbstractEventLoop:
        """Gets or creates an event loop."""
        try:
            current_loop = asyncio.get_event_loop()
            if current_loop.is_closed():
                raise RuntimeError("Event loop is closed.")
            return current_loop
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop

    async def _arequest(self, text: str) -> dict:
        """Makes an async HTTP request to the llama.cpp server."""
        @retry(**self.retry_params)
        async def _make_request():
            timeout = aiohttp.ClientTimeout(
                sock_connect=self.timeout_params['connect'],
                sock_read=self.timeout_params['read']
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.server_endpoint}/v1/embeddings",
                    json={"input": text}
                ) as res:
                    if res.status != 200:
                        raise ValueError(f"Request failed with status code: {res.status}")
                    return await res.json()

        return await _make_request()

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        kwargs = argument.kwargs

        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        new_dim = kwargs.get('new_dim')

        nest_asyncio.apply()
        loop = self._get_event_loop()

        try:
            res = loop.run_until_complete(self._arequest(inp))
        except Exception as e:
            raise ValueError(f"Request failed with error: {str(e)}")

        if new_dim:
            raise NotImplementedError("new_dim is not yet supported")

        if res is not None:
            output = [r["embedding"] for r in res["data"]]
        else:
            output = None
        metadata = {'raw_output': res}

        return [output], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "LlamaCppEmbeddingEngine does not support processed_input."
        argument.prop.prepared_input = argument.prop.entries
