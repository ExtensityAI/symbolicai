import sys
import traceback
from typing import List

from .base import Engine


def full_stack():
    import sys
    import traceback
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[-10:-1]  # last one would be full_stack()
    if exc is not None:                        # i.e. an exception is present
        del stack[-1]                          # remove call of full_stack, the printed exception
                                               # will contain the caught exception caller instead
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
         stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr


class PythonEngine(Engine):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> List[str]:
        code = kwargs['prompt']

        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((code,))

        globals_ = dict(**globals())
        locals_ = dict(**locals())
        rsp = None
        err = None
        try:
            exec(str(code), globals_, locals_)
            rsp = {'globals': globals_, 'locals': locals_}
        except Exception as e:
            err = e
            raise e
        finally:
            output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
            if output_handler:
                if err:
                    stack = full_stack()
                    output_handler(stack)
                else:
                    output_handler(rsp)

        return [rsp]

    def prepare(self, args, kwargs, wrp_params):
        pass
