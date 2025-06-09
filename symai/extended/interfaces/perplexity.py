from ... import core
from ...backend.engines.search.engine_perplexity import SearchResult
from ...symbol import Expression, Symbol


class perplexity(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, query: Symbol, **kwargs) -> SearchResult:
        query = self._to_symbol(query)
        @core.search(query=query.value, **kwargs)
        def _func(_) -> SearchResult:
            pass
        return _func(self)
