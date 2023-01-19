from typing import List
from .base import Engine


class UserInputEngine(Engine):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> List[str]:
        msg = kwargs['prompt']
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((msg,))        
        
        mock = kwargs['mock'] if 'mock' in kwargs else False
        if mock: # mock user input
            print(msg, end='') # print prompt
            rsp = mock
        else:
            rsp = input(msg)
        
        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)
        
        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        wrp_params['prompt'] = wrp_params['processed_input']
