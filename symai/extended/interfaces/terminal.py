from typing import List

from ...symbol import Expression
from ...shellsv import process_command


class terminal(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, command: str = None, **kwargs) -> "terminal":
        return self.sym_return_type(process_command(command, **kwargs))
