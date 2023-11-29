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

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'SYMBOLIC_ENGINE_API_KEY' in wrp_params:
            self.api_key = wrp_params['SYMBOLIC_ENGINE_API_KEY']
            self.client = wa.Client(self.api_key) if len(self.api_key) > 0 else None

    def forward(self, argument):
        queries = argument.prop.processed_input

        if self.client is None:
            self.client = wa.Client(self.api_key) if len(self.api_key) > 0 else None

        queries_ = queries if isinstance(queries, list) else [queries]
        rsp = []

        input_handler = argument.kwargs['input_handler'] if 'input_handler' in argument.kwargs else None
        if input_handler:
            input_handler((queries_,))

        rsp = self.client.query(str(queries[0]))
        rsp = WolframResult(rsp)

        output_handler = argument.kwargs['output_handler'] if 'output_handler' in argument.kwargs else None
        if output_handler:
            output_handler(rsp)

        metadata = {}
        if 'metadata' in argument.kwargs and argument.kwargs['metadata']:
            metadata['kwargs'] = argument.kwargs
            metadata['input']  = queries_
            metadata['output'] = rsp

        return [rsp], metadata

    def prepare(self, argument):
        argument.prop.processed_input = [argument.args[0]]
