from ... import core
from ...backend.engines.search.engine_openai import OpenAISearchResult
from ...symbol import Expression, Symbol


class openai_search(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, query: Symbol, **kwargs) -> OpenAISearchResult:
        query = self._to_symbol(query)

        @core.search(query=query.value, **kwargs)
        def _func(_) -> OpenAISearchResult:
            pass

        return _func(self)
