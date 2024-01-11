from typing import List

from . import core
from . import core_ext
from .symbol import Expression, Symbol
from .components import Function


class Memory(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def store(self, query: str, *args, **kwargs):
        raise NotImplementedError

    def forget(self, query: str, *args, **kwargs):
        raise NotImplementedError

    def recall(self, query: str, *args, **kwargs):
        raise NotImplementedError

    def forward(self, query: str, *args, **kwargs):
        return self.recall(query, *args, **kwargs)


class SlidingWindowListMemory(Memory):
    def __init__(self, window_size: int = 10, max_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
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


class SlidingWindowStringConcatMemory(Memory):
    def __init__(self, token_ratio: float = 0.6, *args, **kwargs):
        super().__init__(**kwargs)
        self._memory: str       = ''
        self.marker: str        = '[--++=|=++--]'
        self.token_ratio: float  = token_ratio

    @core_ext.bind(engine='neurosymbolic', property='max_tokens')
    def max_tokens(self): pass

    def __getstate__(self):
        state = super().__getstate__().copy()
        # Exclude the max_tokens property from being serialized
        state.pop('_max_tokens', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Initialize _max_tokens as None, it should be set again after deserialization
        @core_ext.bind(engine='neurosymbolic', property='max_tokens')
        def _max_tokens(self): pass
        self.max_tokens = _max_tokens

    def history(self):
        return [hist for hist in self._memory.split(self.marker) if hist]

    def drop(self):
        self._memory = ''

    def store(self, query: str):
        # append to string to memory
        self._memory += f'{str(query)}{self.marker}'

    def forget(self, query: Symbol, *args, **kwargs):
        # remove substring from memory
        sym = Symbol(self._memory)
        self._memory = str(sym - query)

    def recall(self, query: str, *args, **kwargs) -> Symbol:
        hist    = self.history()
        memory  = ''
        if len(hist) > 0:
            memory  = '[MEMORY]\n'
            val     = '\n'.join(hist)
            memory += val
        func    = Function(memory)
        func.static_context = self.static_context # TODO: consider dynamic context
        return func(query, *args, **kwargs)


class VectorDatabaseMemory(Memory):
    def __init__(self, enabled: bool = True, top_k: int = 3, index_name: str = 'defaultindex', **kwargs):
        super().__init__(**kwargs)
        self.enabled: bool = enabled
        self.top_k: int    = top_k
        self.index_name    = index_name

    def store(self, query: str , *args, **kwargs):
        if not self.enabled: return

        self.add(Symbol(query).zip(), index_name=self.index_name)

    def recall(self, query: str, *args, **kwargs):
        if not self.enabled: return

        res = self.get(Symbol(query).embed().value, index_top_k=self.top_k, index_name=self.index_name).ast()

        return [v['metadata']['text'] for v in res['matches']]

