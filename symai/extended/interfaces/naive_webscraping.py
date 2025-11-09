from ... import core
from ...backend.engines.webscraping.engine_requests import RequestsResult
from ...symbol import Expression


class naive_webscraping(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, url: str, **kwargs) -> RequestsResult:
        @core.scrape(url=url, **kwargs)
        def _func(_, *_args, **_inner_kwargs) -> RequestsResult:
            # The fallback path may inject debugging kwargs like `error`/`stack_trace`;
            # accept and ignore them so EngineRepository can surface structured failures.
            return None
        return _func(self)
