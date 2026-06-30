from symai import core
from symai.backend.engines.search.engine_serpapi import SerpApiSearchResult
from symai.symbol import Expression, Symbol


class serpapi(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, query: Symbol, **kwargs) -> SerpApiSearchResult:
        query = self._to_symbol(query)

        @core.search(query=query.value, **kwargs)
        def _func(_) -> SerpApiSearchResult:
            pass

        return _func(self)
