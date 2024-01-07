from typing import Any, Optional, List, Union
from .pre_processors import PreProcessor
from .post_processors import PostProcessor


class ProcessorPipeline:
    '''
    Base class for all processors.

    Args:
        processors: A list of processors to be applied to the response.
    '''

    def __init__(self, processors: Optional[List[Union[PreProcessor, PostProcessor]]] = None) -> None:
        self.processors = processors

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.processors is None or len(self.processors) == 0:
            return None

        if isinstance(self.processors[0], PreProcessor):
            assert len(args) == 1, f"Expected 1 argument of type Argument, got {len(args)}"
            processed_input               = ''
            for processor in self.processors:
                assert isinstance(processor, PreProcessor), f"Expected PreProcessor, got {type(processor)}"
                argument         = args[0]
                t                = processor(argument, **kwds)
                processed_input += t if t is not None else ''
            return processed_input

        elif isinstance(self.processors[0], PostProcessor):
            assert len(args) == 2, f"Expected 2 arguments of type Response and Argument, got {len(args)}"
            response, argument = args
            for processor in self.processors:
                assert isinstance(processor, PostProcessor), f"Expected PostProcessor, got {type(processor)}"
                response = processor(response, argument, **kwds)
            return response
