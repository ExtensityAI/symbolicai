from ...base import Engine


class OutputEngine(Engine):
    def __init__(self):
        super().__init__()

    def id(self) -> str:
        return 'output'

    def forward(self, argument):
        kwargs  = argument.kwargs
        expr    = kwargs['expr'] if 'expr' in kwargs else None
        res     = None
        if expr:
            res = expr(*kwargs['args'], **kwargs['kwargs'])

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = expr
            metadata['output'] = res

        return [kwargs], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "OutputEngine does not support processed_input."
