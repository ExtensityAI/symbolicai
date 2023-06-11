import symai as ai


class dall_e(ai.Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sym: ai.Symbol, operation: str = 'create', **kwargs) -> "dall_e":
        sym = self._to_symbol(sym)
        @ai.draw(operation=operation, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(sym))
