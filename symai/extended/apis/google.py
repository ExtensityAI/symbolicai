from ...symbol import Symbol, Expression
from ... import core


class google(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, query: Symbol, **kwargs) -> "google":
        query = self._to_symbol(query)
        @core.search(query=query.value, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
