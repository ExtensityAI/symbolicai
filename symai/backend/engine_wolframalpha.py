from .settings import SYMAI_CONFIG
from IPython.utils import io
from typing import Tuple, List
from .base import Engine
import wolframalpha as wa


class WolframAlphaEngine(Engine):
    def __init__(self):
        super().__init__()
        config = SYMAI_CONFIG         
        self.api_key = config['SYMBOLIC_ENGINE_API_KEY']
        self.client = wa.Client(self.api_key) if len(self.api_key) > 0 else None

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'SYMBOLIC_ENGINE_API_KEY' in wrp_params:
            self.api_key = wrp_params['SYMBOLIC_ENGINE_API_KEY']
            self.client = wa.Client(self.api_key) if len(self.api_key) > 0 else None

    def forward(self, queries: List[str], *args, **kwargs) -> List[str]:
        assert self.client is not None, "WolframAlpha API key is not set."
        
        queries_ = queries if isinstance(queries, list) else [queries]
        rsp = []
        
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((queries_,))
        
        rsp = self.client.query(str(queries[0]))
        
        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        wrp_params['queries'] = [args[0]]
