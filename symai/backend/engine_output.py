from typing import List
from .base import Engine


class OutputEngine(Engine):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> List[str]:
        expr = kwargs['expr'] if 'expr' in kwargs else None
        res = None
        if expr:
            def input_handler(vals):
                kwargs['input'] = vals
            kwargs['kwargs']['input_handler'] = input_handler
            res = expr(*kwargs['args'], **kwargs['kwargs'])
        
        kwargs['output'] = res
        handler = kwargs['handler'] if 'handler' in kwargs else None
        if handler:
            handler(kwargs)
        
        return [kwargs]
    
    def prepare(self, args, kwargs, wrp_params):
        pass
