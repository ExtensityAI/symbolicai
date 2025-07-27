from ... import core
from ...backend.engines.webscraping.engine_requests import RequestsResult
from ...symbol import Expression


class naive_webscraping(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, url: str, **kwargs) -> RequestsResult:
        @core.scrape(url=url, **kwargs)
        def _func(_) -> RequestsResult:
            pass
        return _func(self)
