import symai as ai


class file(ai.Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, path: ai.Symbol, **kwargs) -> "file":
        path = self._to_symbol(path)
        @ai.opening(path=path.value, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
