from symai import *


class Crawler(Expression):
    def __init__(self, filters: List[Expression] = []):
        super().__init__()
        filters = filters if isinstance(filters, List) or isinstance(filters, tuple) else [filters]
        self.data_stream = Stream(Sequence(
            Clean(),
            *filters
        ))
        
    def forward(self, url: str, **kwargs) -> Symbol:
        res = self.fetch(url=url)
        vals = list(self.data_stream(res, **kwargs))
        return Symbol(vals)
