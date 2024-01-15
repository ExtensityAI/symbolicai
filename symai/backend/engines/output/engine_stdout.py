from ...base import Engine


class OutputEngine(Engine):
    def __init__(self):
        super().__init__()

    def id(self) -> str:
        return 'output'

    def forward(self, argument):
        expr, processed, args, kwargs  = argument.prop.prepared_input
        res    = None
        args   = [] if args is None else args
        kwargs = {} if kwargs is None else kwargs
        if expr:
            if processed:
                res = expr(processed, *args, **kwargs)
            else:
                res = expr(*args, **kwargs)

        metadata = {}
        result   = {
            'result': res,
            'processed': processed,
            'args': args,
            'kwargs': kwargs
        }

        return [result], metadata

    def prepare(self, argument):
        argument.prop.prepared_input = argument.prop.expr, argument.prop.processed_input, argument.prop.args, argument.prop.kwargs
