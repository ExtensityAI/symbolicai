import logging
import os
import time
from abc import ABC
from typing import Any, List


class Engine(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.verbose    = False
        self.logging    = False
        self.log_level  = logging.DEBUG
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
        start_time = time.time()
        res, metadata = self.forward(*args, **kwds)
        req_time = time.time() - start_time
        metadata['time'] = req_time
        if self.time_clock:
            print(f"{kwds['func']}: {req_time} sec")
        log['Output'] = res
        if self.verbose:
            view   = {k: v for k, v in list(log['Input'].items()) if k != 'self' and k != 'func' and k != 'args'}
            input_ = f"{str(log['Input']['self'])[:50]}, {str(log['Input']['func'])}, {str(view)}"
            print(input_[:150], str(log['Output'])[:100])
        if self.logging:
            self.logger.log(self.log_level, log)
        return res, metadata

    def preview(self, wrp_params):
        return str(wrp_params['prompts']), {}

    def forward(self, *args: Any, **kwds: Any) -> List[str]:
        raise NotADirectoryError()

    def prepare(self, args, kwargs, wrp_params):
        raise NotImplementedError()

    def command(self, wrp_params):
        if 'verbose' in wrp_params:
            self.verbose = wrp_params['verbose']
        if 'logging' in wrp_params:
            self.logging = wrp_params['logging']
        if 'log_level' in wrp_params:
            self.log_level = wrp_params['log_level']
        if 'time_clock' in wrp_params:
            self.time_clock = wrp_params['time_clock']
