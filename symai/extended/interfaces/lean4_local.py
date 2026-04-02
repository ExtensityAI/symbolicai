from ... import core
from ...backend.engines.formal.engine_lean4_local import LeanResult
from ...symbol import Expression


class lean4_local(Expression):
    def __call__(self, expr, config=None, **kwargs) -> LeanResult:
        @core.formal(config=config or {}, **kwargs)
        def _func(_, expr):
            pass

        return _func(self, expr)
