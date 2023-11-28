import logging
import os
import time

from abc import ABC
from typing import Any, List, Tuple

from ..collect import CollectionRepository, rec_serialize


ENGINE_UNREGISTERED = '<UNREGISTERED/>'


class PreviewSymbol(ABC):
    '''
    A Symbol subclass that can be used to create new Symbol subclasses to store metadata.
    '''
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

    def __call__(self, argument: Any) -> Tuple[List[str], dict]:
        log = {
            'Input': {
                'self': self,
                'args': argument.args,
                **argument.kwargs
            }
        }
        start_time = time.time()
        res, metadata = self.forward(argument)
        req_time = time.time() - start_time
        metadata['time'] = req_time
        if self.time_clock:
            print(f"{argument.prop.func}: {req_time} sec")
        log['Output'] = res
        if self.verbose:
            view   = {k: v for k, v in list(log['Input'].items()) if k != 'self' and k != 'func' and k != 'args'}
            input_ = f"{str(log['Input']['self'])[:50]}, {str(log['Input']['func'])}, {str(view)}"
            print(input_[:150], str(log['Output'])[:100])
        if self.logging:
            self.logger.log(self.log_level, log)
        if     str(self) == 'GPTXChatEngine' \
            or str(self) == 'GPTXCompletionEngine' \
            or str(self) == 'SerpApiEngine' \
            or str(self) == 'WolframAlphaEngine' \
            or str(self) == 'SeleniumEngine' \
            or str(self) == 'OCREngine':
            self.collection.add(
                forward={'args': rec_serialize(argument.args), 'kwds': rec_serialize(argument.kwargs)},
                engine=str(self),
                metadata={
                    'time': req_time,
                    'data': rec_serialize(metadata),
                    'argument': rec_serialize(argument)
                }
            )
        return res, metadata

    def id(self) -> str:
        return ENGINE_UNREGISTERED

    def preview(self, argument):
        return PreviewSymbol(argument), {}

    def forward(self, *args: Any, **kwds: Any) -> List[str]:
        raise NotADirectoryError()

    def prepare(self, argument):
        raise NotImplementedError()

    def command(self, argument):
        if argument.prop.verbose:
            self.verbose = argument.prop.verbose
        if argument.prop.logging:
            self.logging = argument.prop.logging
        if argument.prop.log_level:
            self.log_level = argument.prop.log_level
        if argument.prop.time_clock:
            self.time_clock = argument.prop.time_clock

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()

