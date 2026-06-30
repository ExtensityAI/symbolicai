from symai import core
from symai.backend.engines.search.engine_parallel import ParallelExtractResult, ParallelSearchResult
from symai.symbol import Expression, Symbol


class parallel(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def search(self, query: Symbol, **kwargs) -> ParallelSearchResult:
        query = self._to_symbol(query)

        @core.search(query=query.value, **kwargs)
        def _func(_) -> ParallelSearchResult:
            pass

        return _func(self)

    def scrape(self, url: str, **kwargs) -> ParallelExtractResult:
        symbol = self._to_symbol(url)
        options = dict(kwargs)
        options.pop("query", None)
        options["url"] = symbol.value

        @core.search(query="", **options)
        def _func(_, *_args, **_inner_kwargs) -> ParallelExtractResult:
            return None

        return _func(self)
