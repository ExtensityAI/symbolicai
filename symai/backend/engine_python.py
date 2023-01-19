from typing import List
from .base import Engine
from symai import *


class PythonEngine(Engine):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> List[str]:
        code = kwargs['prompt']
        
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((code,))
        
        globals_ = dict(**globals())
        locals_ = dict(**locals())
        exec(str(code), globals_, locals_)
        rsp = {'globals': globals_, 'locals': locals_}
        
        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)
        
        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        pass
