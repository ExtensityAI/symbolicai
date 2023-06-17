from itertools import chain
from typing import List

from .symbol import Symbol, Expression


class Memory(Expression):
    def __init__(self):
        super().__init__()

    def store(self, query: str, *args, **kwargs):
        raise NotImplementedError

    def forget(self, query: str, *args, **kwargs):
        raise NotImplementedError

    def recall(self, query: str, *args, **kwargs):
        raise NotImplementedError

    def forward(self, query: str, *args, **kwargs):
        return self.recall(query, *args, **kwargs)


class SlidingWindowListMemory(Memory):
    def __init__(self, window_size: int = 10, max_size: int = 1000):
        super().__init__()
        self._memory: List[str] = []
        self._window_size: int  = window_size
        self._max_size: int     = max_size

    def store(self, query: str, *args, **kwargs):
        self._memory.append(query)
        if len(self._memory) > self._max_size:
            self._memory = self._memory[-self._max_size:]

    def forget(self, query: Symbol, *args, **kwargs):
        self._memory.remove(query)

    def recall(self, *args, **kwargs):
        return self._memory[-self._window_size:]


class VectorDatabaseMemory(Memory):
    def __init__(self, enabled: bool = True, top_k: int = 3):
        super().__init__()
        self.enabled: bool = enabled
        self.top_k: int    = top_k

    def store(self, query: str , *args, **kwargs):
        if not self.enabled: return

        self.add(Symbol(query).zip())

    def recall(self, query: str, *args, **kwargs):
        if not self.enabled: return

        res = self.get(Symbol(query).embed().value, index_top_k=self.top_k).ast()
        *res, = chain.from_iterable([v['metadata']['text'] for v in res['matches']])

        return res

