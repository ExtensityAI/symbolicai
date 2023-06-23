from ...symbol import Expression
from ... import core


class console(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> "console":
        kwargs['handler'] = lambda x: print(*x['args'])
        @core.output(**kwargs)
        def _func(_, *args):
            pass
        return self._sym_return_type(_func(self, *args))
