import symai as ai


class python(ai.Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, expr: str, **kwargs) -> "python":
        sym = self._to_symbol(expr)
        @ai.execute(**kwargs)
        def _func(_):
            pass
        return _func(sym)
