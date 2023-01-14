from typing import Tuple, List
from .base import Engine


class UserInputEngine(Engine):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> List[str]:
        msg = kwargs['prompt']
        
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((msg,))        
        
        rsp = input(msg)
        
        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)
        
        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        pass