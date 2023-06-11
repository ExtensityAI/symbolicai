import symai as ai


class pinecone(ai.Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, stmt: str, operation: str = "search", **kwargs) -> "pinecone":
        stmt = self._to_symbol(stmt)
        @ai.index(prompt=stmt.value, operation=operation, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
