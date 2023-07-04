from typing import Optional
from datetime import datetime
from .. import Symbol, SlidingWindowStringConcatMemory
from ..components import Indexer

class Conversation(SlidingWindowStringConcatMemory):
    def __init__(self, init: Optional[str] = None, auto_print: bool = True, token_ratio: float = 0.6, *args, **kwargs):
        super().__init__(token_ratio)
        self.indexer = Indexer()
        self._index = self.indexer()
        self.token_ratio = token_ratio
        self.auto_print = auto_print
        if init is not None:
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
            val = str(f"[SYSTEM::{timestamp}] {str(init)}\n")
            self.store(val, *args, **kwargs)

    def index(self, file_path: str):
        return self._index(file_path)

    def forward(self, query: str, *args, **kwargs):
        query = self._to_symbol(query)
        # get timestamp in string format
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
        # append to string to memory
        val = str(f"[USER::{timestamp}] {str(query)}\n")
        self.store(val, *args, **kwargs)
        history = Symbol(f'[HISTORY]========\n{self._memory}\n[END_HISTORY]========\n')
        res = self.recall(query, payload=history, *args, **kwargs)
        val = str(f"[ASSISTANT::{timestamp}] {str(res)}\n")
        self.store(val, *args, **kwargs)
        if self.auto_print:
            print(res)
        return res
