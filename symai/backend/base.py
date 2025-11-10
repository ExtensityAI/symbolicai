import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from ..collect import CollectionRepository, rec_serialize
from ..utils import UserMessage
from .settings import HOME_PATH

ENGINE_UNREGISTERED = '<UNREGISTERED/>'

COLLECTION_LOGGING_ENGINES = {
    'GPTXChatEngine',
    'GPTXCompletionEngine',
    'SerpApiEngine',
    'WolframAlphaEngine',
    'SeleniumEngine',
    'OCREngine',
}

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

        self._trigger_input_handlers(argument)

        res, metadata = self.forward(argument)

        req_time = time.time() - start_time
        metadata['time'] = req_time
        if self.time_clock:
            UserMessage(f"{argument.prop.func}: {req_time} sec")
        log['Output'] = res
        if self.verbose:
            view   = {k: v for k, v in list(log['Input'].items()) if k != 'self'}
            input_ = f"{str(log['Input']['self'])[:50]}, {argument.prop.func!s}, {view!s}"
            UserMessage(f"{input_[:150]} {str(log['Output'])[:100]}")
        if self.logging:
            self.logger.log(self.log_level, log)

        if str(self) in COLLECTION_LOGGING_ENGINES:
            self._record_collection_entry(argument, metadata, req_time)

        self._trigger_output_handlers(argument, res, metadata)
        return res, metadata

    def _trigger_input_handlers(self, argument: Any) -> None:
        instance_metadata = getattr(argument.prop.instance, '_metadata', None)
        if instance_metadata is not None:
            input_handler = getattr(instance_metadata, 'input_handler', None)
            if input_handler is not None:
                input_handler((argument.prop.processed_input, argument))
        argument_handler = argument.prop.input_handler
        if argument_handler is not None:
            argument_handler((argument.prop.processed_input, argument))

    def _trigger_output_handlers(self, argument: Any, result: Any, metadata: dict | None) -> None:
        instance_metadata = getattr(argument.prop.instance, '_metadata', None)
        if instance_metadata is not None:
            output_handler = getattr(instance_metadata, 'output_handler', None)
            if output_handler:
                output_handler(result)
        argument_handler = argument.prop.output_handler
        if argument_handler:
            argument_handler((result, metadata))

    def _record_collection_entry(self, argument: Any, metadata: dict, req_time: float) -> None:
        self.collection.add(
            forward={'args': rec_serialize(argument.args), 'kwds': rec_serialize(argument.kwargs)},
            engine=str(self),
            metadata={
                'time': req_time,
                'data': rec_serialize(metadata),
                'argument': rec_serialize(argument)
            }
        )

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
            self._trigger_input_handlers(arg)

        results, metadata_list = self._execute_batch(arguments)

        total_time = time.time() - start_time
        if self.time_clock:
            UserMessage(f"Total execution time: {total_time} sec")

        return self._prepare_batch_results(arguments, results, metadata_list, total_time)

    def _execute_batch(self, arguments: list[Any]) -> tuple[list[Any], list[dict | None]]:
        try:
            return self.forward(arguments)
        except Exception as error:
            return [error] * len(arguments), [None] * len(arguments)

    def _prepare_batch_results(
        self,
        arguments: list[Any],
        results: list[Any],
        metadata_list: list[dict | None],
        total_time: float,
    ) -> list[tuple[Any, dict | None]]:
        return_list = []
        for arg, result, metadata in zip(arguments, results, metadata_list, strict=False):
            if metadata is not None:
                metadata['time'] = total_time / len(arguments)

            self._trigger_output_handlers(arg, result, metadata)
            return_list.append((result, metadata))
        return return_list

    def forward(self, _arguments: list[Any]) -> tuple[list[Any], list[dict]]:
        msg = "Subclasses must implement forward method"
        UserMessage(msg)
        raise NotImplementedError(msg)
