import logging
import os
import time
import json
from abc import ABC
from typing import Any, List
from ..collect import CollectionRepository, rec_serialize
from ..symbol import Symbol


class PreviewSymbol(Symbol):
    def __init__(self, params: dict):
        super().__init__(params['prompts'])
        self.params = params


class Engine(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.verbose    = False
        self.logging    = False
        self.log_level  = logging.DEBUG
        self.time_clock = False
        self.collection = CollectionRepository()
        self.collection.connect()
        # create formatter
        os.makedirs('outputs', exist_ok=True)
        logging.basicConfig(filename="outputs/engine.log", filemode="a", format='%(asctime)s %(name)s %(levelname)s %(message)s')
        self.logger     = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # logging to console
        stream          = logging.StreamHandler()
        streamformat    = logging.Formatter("%(asctime)s %(message)s")
        stream.setLevel(logging.INFO)
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
        if 'metadata' not in kwds:
            kwds['metadata'] = True
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
        self.collection.add(
            forward={'args': rec_serialize(args), 'kwds': rec_serialize(kwds)},
            engine=str(self),
            metadata={
                'time': req_time,
                'data': rec_serialize(metadata)
            }
        )
        return res, metadata

    def preview(self, wrp_params):
        return PreviewSymbol(wrp_params), {}

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

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()

