import json

from box import Box

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Symbol

try:
    import wolframalpha as wa
except:
    wa = None


class WolframResult(Symbol):
    def __init__(self, value) -> None:
        super().__init__(value)
        self.raw = Box(value)
        self._value = value


class WolframAlphaEngine(Engine):
    def __init__(self):
        super().__init__()
        self.config  = SYMAI_CONFIG
        self.api_key = self.config['SYMBOLIC_ENGINE_API_KEY']
        self.client  = None

    def id(self) -> str:
        if self.config['SYMBOLIC_ENGINE_API_KEY']:
            if wa is None:
                print('WolframAlpha is not installed. Please install it with `pip install symbolicai[wolframalpha]`')
            return 'symbolic'
        return super().id() # default to unregistered

    def command(self, argument):
        super().command(argument.kwargs)
        if 'SYMBOLIC_ENGINE_API_KEY' in argument.kwargs:
            self.api_key = argument.kwargs['SYMBOLIC_ENGINE_API_KEY']
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
