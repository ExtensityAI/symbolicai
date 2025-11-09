import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from ..collect import CollectionRepository, rec_serialize
from ..utils import CustomUserWarning
from .settings import HOME_PATH

ENGINE_UNREGISTERED = '<UNREGISTERED/>'

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
        __root_dir__  = HOME_PATH
        __root_dir__.mkdir(parents=True, exist_ok=True)
        __file_path__ = __root_dir__ / "engine.log"
        logging.basicConfig(filename=__file_path__, filemode="a", format='%(asctime)s %(name)s %(levelname)s %(message)s')
        self.logger     = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # logging to console
        stream          = logging.StreamHandler()
        streamformat    = logging.Formatter("%(asctime)s %(message)s")
        stream.setLevel(logging.INFO)
        stream.setFormatter(streamformat)
        self.logger.addHandler(stream)

    def __call__(self, argument: Any) -> tuple[list[str], dict]:
        log = {
            'Input': {
                'self': self,
                'args': argument.args,
                **argument.kwargs
            }
        }
        start_time = time.time()

        # check for global object based input handler
        if hasattr(argument.prop.instance, '_metadata') and hasattr(argument.prop.instance._metadata, 'input_handler'):
            input_handler = argument.prop.instance._metadata.input_handler if hasattr(argument.prop.instance._metadata, 'input_handler') else None
            if input_handler is not None:
                input_handler((argument.prop.processed_input, argument))
        # check for kwargs based input handler
        if argument.prop.input_handler is not None:
            argument.prop.input_handler((argument.prop.processed_input, argument))

        # execute the engine
        res, metadata = self.forward(argument)

        # compute time
        req_time = time.time() - start_time
        metadata['time'] = req_time
        if self.time_clock:
            CustomUserWarning(f"{argument.prop.func}: {req_time} sec")
        log['Output'] = res
        if self.verbose:
            view   = {k: v for k, v in list(log['Input'].items()) if k != 'self'}
            input_ = f"{str(log['Input']['self'])[:50]}, {argument.prop.func!s}, {view!s}"
            CustomUserWarning(f"{input_[:150]} {str(log['Output'])[:100]}")
        if self.logging:
            self.logger.log(self.log_level, log)

        # share data statistics with the collection repository
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

        # check for global object based output handler
        if hasattr(argument.prop.instance, '_metadata') and hasattr(argument.prop.instance._metadata, 'output_handler'):
            output_handler = argument.prop.instance._metadata.output_handler if hasattr(argument.prop.instance._metadata, 'output_handler') else None
            if output_handler:
                output_handler(res)
        # check for kwargs based output handler
        if argument.prop.output_handler:
            argument.prop.output_handler((res, metadata))
        return res, metadata

    def id(self) -> str:
        return ENGINE_UNREGISTERED

    def preview(self, argument):
        # Used here to avoid backend.base <-> symbol circular import.
        from ..symbol import ( # noqa
            Symbol,
        )
        class Preview(Symbol):
            def __repr__(self) -> str:
                '''
                Get the representation of the Symbol object as a string.

                Returns:
                    str: The representation of the Symbol object.
                '''
                return str(self.value.prop.prepared_input)

            def prepared_input(self):
                return self.value.prop.prepared_input

        return Preview(argument), {}

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> list[str]:
        raise NotADirectoryError

    @abstractmethod
    def prepare(self, argument):
        raise NotImplementedError

    def command(self, *_args, **kwargs):
        if kwargs.get('verbose'):
            self.verbose = kwargs['verbose']
        if kwargs.get('logging'):
            self.logging = kwargs['logging']
        if kwargs.get('log_level'):
            self.log_level = kwargs['log_level']
        if kwargs.get('time_clock'):
            self.time_clock = kwargs['time_clock']

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        '''
        Get the representation of the Symbol object as a string.

        Returns:
            str: The representation of the Symbol object.
        '''
        # class with full path
        class_ = self.__class__.__module__ + '.' + self.__class__.__name__
        hex_   = hex(id(self))
        return f'<class {class_} at {hex_}>'


class BatchEngine(Engine):
    def __init__(self):
        super().__init__()
        self.time_clock = True
        self.allows_batching = True

    def __call__(self, arguments: list[Any]) -> list[tuple[Any, dict]]:
        start_time = time.time()
        for arg in arguments:
            if hasattr(arg.prop.instance, '_metadata') and hasattr(arg.prop.instance._metadata, 'input_handler'):
                input_handler = getattr(arg.prop.instance._metadata, 'input_handler', None)
                if input_handler is not None:
                    input_handler((arg.prop.processed_input, arg))
            if arg.prop.input_handler is not None:
                arg.prop.input_handler((arg.prop.processed_input, arg))

        try:
            results, metadata_list = self.forward(arguments)
        except Exception as e:
            results = [e] * len(arguments)
            metadata_list = [None] * len(arguments)

        total_time = time.time() - start_time
        if self.time_clock:
            CustomUserWarning(f"Total execution time: {total_time} sec")

        return_list = []

        for arg, result, metadata in zip(arguments, results, metadata_list, strict=False):
            if metadata is not None:
                metadata['time'] = total_time / len(arguments)

            if hasattr(arg.prop.instance, '_metadata') and hasattr(arg.prop.instance._metadata, 'output_handler'):
                output_handler = getattr(arg.prop.instance._metadata, 'output_handler', None)
                if output_handler:
                    output_handler(result)
            if arg.prop.output_handler:
                arg.prop.output_handler((result, metadata))

            return_list.append((result, metadata))
        return return_list

    def forward(self, _arguments: list[Any]) -> tuple[list[Any], list[dict]]:
        msg = "Subclasses must implement forward method"
        CustomUserWarning(msg)
        raise NotImplementedError(msg)
