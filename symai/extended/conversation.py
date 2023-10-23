import os
from datetime import datetime
from typing import Any, Callable, Optional, List

from ..components import Indexer
from ..memory import SlidingWindowStringConcatMemory
from ..symbol import Symbol


class CodeFormatter:
    def __call__(self, value: str, *args: Any, **kwds: Any) -> Any:
        # extract code from chat conversations or ```<language>\n{code}\n``` blocks
        return Symbol(value).extract('Only extract code without ``` block markers or chat conversations')


class Conversation(SlidingWindowStringConcatMemory):
    def __init__(self, init:     Optional[str] = None,
                 file_link:      Optional[List[str]] = None,
                 index_name:     str           = Indexer.DEFAULT,
                 auto_print:     bool          = True,
                 token_ratio:    float         = 0.6, *args, **kwargs):
        super().__init__(token_ratio)
        self.token_ratio = token_ratio
        self.auto_print  = auto_print
        if file_link is not None and type(file_link) is str:
            file_link = [file_link]
        self.file_link   = file_link

        if init is not None:
            self.store_system_message(init, *args, **kwargs)
        if file_link is not None:
            for fl in file_link:
                self.store_file(fl, *args, **kwargs)
        self.indexer = Indexer(index_name=index_name)
        self._index  = self.indexer()

    def store_system_message(self, message: str, *args, **kwargs):
        val = f"[SYSTEM::INSTRUCTION] <<<\n{str(message)}\n>>>\n"
        self.store(val, *args, **kwargs)

    def store_file(self, file_path: str, *args, **kwargs):
        if not os.path.exists(file_path):
            return
        # read in file
        with open(file_path, 'r') as file:
            content = file.read()
        val = f"[DATA::{file_path}] <<<\n{str(content)}\n>>>\n"
        self.store(val, *args, **kwargs)

    def commit(self, target_file: str = None, formatter: Optional[Callable] = None):
        if target_file is not None and type(target_file) is str:
            file_link = target_file
        else:
            file_link = self.file_link if self.file_link is not None and type(self.file_link) is str else None
        if file_link is not None:
            # if file extension is .py, then format code
            format_ = formatter
            formatter = CodeFormatter() if format_ is None and file_link.endswith('.py') else formatter
            val = self.value
            if formatter is not None:
                val = formatter(val)
            # if file does not exist, create it
            with open(file_link, 'w') as file:
                file.write(str(val))
        else:
            raise Exception('File link is not set or a set of files.')

    def save(self, path: str, replace: bool = False) -> Symbol:
        return Symbol(self._memory).save(path, replace=replace)

    def index(self, file_path: str):
        return self._index(file_path)

    def forward(self, query: str, *args, **kwargs):
        query = self._to_symbol(query)
        # get timestamp in string format
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
        # append to string to memory
        val = str(f"[USER::{timestamp}] <<<\n{str(query)}\n>>>\n")
        self.store(val, *args, **kwargs)
        history = Symbol(f'[HISTORY] <<<\n{self._memory}\n>>>\n')
        if 'payload' in kwargs:
            history =  f'{history}\n{kwargs["payload"]}'
            del kwargs['payload']
        res = self.recall(query, payload=history, *args, **kwargs)
        self.value = res.value # save last response
        val = str(f"[ASSISTANT::{timestamp}] <<<\n{str(res)}\n>>>\n")
        self.store(val, *args, **kwargs)
        if self.auto_print:
            print(res)
        return res

    def __repr__(self):
            """Get the representation of the Symbol object as a string.

            Returns:
                str: The representation of the Symbol object.
            """
            return str(self.value)
