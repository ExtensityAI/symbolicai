from symai import core
from symai.backend.engines.search.engine_perplexity import PerplexitySearchResult
from symai.symbol import Expression, Symbol


class perplexity(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, query: Symbol, **kwargs) -> PerplexitySearchResult:
        query = self._to_symbol(query)

        @core.search(query=query.value, **kwargs)
        def _func(_) -> PerplexitySearchResult:
            pass

        return _func(self)
