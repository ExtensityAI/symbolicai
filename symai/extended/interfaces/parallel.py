from ... import core
from ...backend.engines.search.engine_parallel import ExtractResult, SearchResult
from ...symbol import Expression, Symbol


class parallel(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def search(self, query: Symbol, **kwargs) -> SearchResult:
        query = self._to_symbol(query)

        @core.search(query=query.value, **kwargs)
        def _func(_) -> SearchResult:
            pass

        return _func(self)

    def scrape(self, url: str, **kwargs) -> ExtractResult:
        symbol = self._to_symbol(url)
        options = dict(kwargs)
        options.pop("query", None)
        options["url"] = symbol.value

        @core.search(query="", **options)
        def _func(_, *_args, **_inner_kwargs) -> ExtractResult:
            return None

        return _func(self)
