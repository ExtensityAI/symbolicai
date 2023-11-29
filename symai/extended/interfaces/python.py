from ... import core
from ...symbol import Expression
from ...backend.engines.execute.engine_python import PythonResult


class python(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, expr: str, **kwargs) -> PythonResult:
        sym = self._to_symbol(expr)
        @core.execute(**kwargs)
        def _func(_) -> PythonResult:
            pass
        return _func(sym)
