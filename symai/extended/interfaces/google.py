from ... import core
from ...symbol import Expression, Symbol
from ...backend.engine_google import SearchResult


class google(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, query: Symbol, **kwargs) -> SearchResult:
        query = self._to_symbol(query)
        @core.search(query=query.value, **kwargs)
        def _func(_) -> SearchResult:
            pass
        return _func(self)
