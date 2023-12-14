import json

from box import Box

from typing import Optional

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Result

try:
    import wolframalpha as wa
except:
    wa = None


class WolframResult(Result):
    def __init__(self, value) -> None:
        super().__init__(value)
        self.raw = Box(value)
        self._value = value


class WolframAlphaEngine(Engine):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.config  = SYMAI_CONFIG
        self.api_key = self.config['SYMBOLIC_ENGINE_API_KEY'] if api_key is None else api_key
        self.client  = None

    def id(self) -> str:
        if self.config['SYMBOLIC_ENGINE_API_KEY']:
            if wa is None:
                print('WolframAlpha is not installed. Please install it with `pip install symbolicai[wolframalpha]`')
            return 'symbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'SYMBOLIC_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['SYMBOLIC_ENGINE_API_KEY']
            self.client  = wa.Client(self.api_key) if len(self.api_key) > 0 else None

    def forward(self, argument):
        queries = argument.prop.prepared_input

        if self.client is None:
            self.client = wa.Client(self.api_key) if len(self.api_key) > 0 else None

        queries_ = queries if isinstance(queries, list) else [queries]
        rsp = []

        rsp = self.client.query(queries)
        rsp = WolframResult(rsp)

        metadata = {}
        if 'metadata' in argument.kwargs and argument.kwargs['metadata']:
            metadata['kwargs'] = argument.kwargs
            metadata['input']  = queries_
            metadata['output'] = rsp

        return [rsp], metadata

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)
