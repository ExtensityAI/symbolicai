from ... import core
from ...symbol import Expression
from ...backend.engines.index.engine_pinecone import PineconeResult


class pinecone(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, stmt: str, operation: str = "search", **kwargs) -> PineconeResult:
        stmt = self._to_symbol(stmt)
        @core.index(prompt=stmt.value, operation=operation, **kwargs)
        def _func(_) -> PineconeResult:
            pass
        return _func(self)
