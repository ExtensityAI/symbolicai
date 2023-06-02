from typing import List, Optional
import symai as ai


class Memory(ai.Expression):
    def __init__(self):
        super().__init__()

    def _check_and_cast_expr(self, sym) -> ai.Expression:
        # check type and cast
        if isinstance(sym, ai.Expression):
            return sym
        return ai.Expression(sym)

    def store(self, sym, *args, **kwargs):
        raise NotImplementedError
    
    def forget(self, sym, *args, **kwargs):
        raise NotImplementedError
    
    def recall(self, sym, *args, **kwargs):
        raise NotImplementedError

    def forward(self, sym, *args, **kwargs):
        sym = self._check_and_cast_expr(sym)
        return self.recall(sym, *args, **kwargs)
    

class SlidingWindowListMemory(Memory):
    def __init__(self, window_size: int = 10, max_size: int = 1000):
        super().__init__()
        self._memory: List[ai.Symbol] = []
        self._window_size: int = window_size
        self._max_size: int = max_size

    def store(self, sym, *args, **kwargs):
        self._memory.append(sym)
        if len(self._memory) > self._max_size:
            self._memory = self._memory[-self._max_size:]

    def forget(self, sym, *args, **kwargs):
        self._memory.remove(sym)
    
    def recall(self, sym = None, *args, **kwargs):
        return self._memory[-self._window_size:]
    

class VectorDatabaseMemory(Memory):
    def __init__(self, enabled: bool = True, top_k: int = 3):
        super().__init__()
        self.enabled: bool = enabled
        self.top_k: int = top_k

    def store(self, sym, *args, **kwargs):
        if not self.enabled:
            return
        sym = self._check_and_cast_expr(sym)
        self.add(sym.zip())
    
    def recall(self, sym, *args, **kwargs):
        if not self.enabled:
            return []
        sym = self._check_and_cast_expr(sym)
        res = self.get(sym.embed().value).ast()
        res = [v['id'] for v in res['matches'][:self.top_k]]
        return res
