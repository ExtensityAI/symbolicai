import symai as ai


class google(ai.Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, query: ai.Symbol, **kwargs) -> "google":
        query = self._to_symbol(query)
        @ai.search(query=query.value, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
