from symai import core
from symai.backend.engines.formal.engine_lean4_local import LeanResult
from symai.symbol import Expression


class lean4_local(Expression):
    def __call__(self, expr, config=None, **kwargs) -> LeanResult:
        @core.formal(config=config or {}, **kwargs)
        def _func(_, expr):
            pass

        return _func(self, expr)
