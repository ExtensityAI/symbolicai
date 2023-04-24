import os
import logging
import time
from abc import ABC
from typing import Any, List


class Engine(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.verbose = False
        self.logging = False
        self.time_clock = False
        # create formatter
        os.makedirs('outputs', exist_ok=True)
        logging.basicConfig(filename="outputs/engine.log", filemode="a", format='%(asctime)s %(name)s %(levelname)s %(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # logging to console
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        streamformat = logging.Formatter("%(asctime)s %(message)s")
        stream.setFormatter(streamformat)
        self.logger.addHandler(stream)

    def __call__(self, *args: Any, **kwds: Any) -> List[str]:
        log = {
            'Input': {
                'self': self,
                'args': args,
                **kwds
            }
        }
        start_time = None
        if self.time_clock:
            start_time = time.time()
        res = self.forward(*args, **kwds)
        if start_time is not None:
            print(f"{kwds['func']}: {(time.time() - start_time)} sec")
        log['Output'] = res
        if self.verbose:
            view = {k: v for k, v in list(log['Input'].items()) if k != 'self' and k != 'func' and k != 'args'}
            input_ = f"{str(log['Input']['self'])[:50]}, {str(log['Input']['func'])}, {str(view)}"
            print(input_[:150], str(log['Output'])[:100])
        if self.logging:
            self.logger.debug(log)
        return res
    
    def preview(self, wrp_params):
        return str(wrp_params['prompts'])
    
    def forward(self, *args: Any, **kwds: Any) -> List[str]:
        raise NotADirectoryError()
    
    def prepare(self, args, kwargs, wrp_params):
        raise NotImplementedError()
    
    def command(self, wrp_params):
        if 'verbose' in wrp_params:
            self.verbose = wrp_params['verbose']
        if 'logging' in wrp_params:
            self.logging = wrp_params['logging']
        if 'time_clock' in wrp_params:
            self.time_clock = wrp_params['time_clock']
