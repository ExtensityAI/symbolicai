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
    """PythonEngine executes Python code dynamically while managing the code execution
    context using provided global and local namespaces. This engine encapsulates the
    execution process, handling both successful execution and exception scenarios.

    Usage:
    - Instantiate the PythonEngine.
    - Call the forward method with the desired Python code and context variables.

    The forward method expects a 'prompt' containing the executable code and can also
    accept 'globals', 'locals', 'input_handler', 'output_handler', and 'metadata' as keyword
    arguments to manage the execution context and handle input/output operations.

    The Python code should contain a 'run' function which serves as the entry point. The
    run function can accept any arguments (*args, **kwargs) and return any type of value.
    The code execution result is encapsulated in 'res', which is then managed by the
    local or global namespace.

    Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments containing 'prompt' and context variables.

    Returns:
        A list containing the results of the code execution within a context dict and a
        metadata dict capturing execution details.

    Example:

        engine = PythonEngine()
        code = '''
        def run(*args, **kwargs):
            return "Hello, " + args[0]
        res = run(value)
        '''
        results, metadata = engine.forward(prompt=code, globals={}, locals={}, value='World')

    Note:
    The 'run' function must be defined, and its execution result must be assigned to 'res'.
    In case of an exception, the engine will provide a full traceback to help with debugging.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> List[str]:
        code = kwargs['prompt']

        globals_      = kwargs['globals'] if 'globals' in kwargs else {}
        locals_       = kwargs['locals']  if 'locals'  in kwargs else {}
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((code,))

        rsp = None
        err = None
        try:
            exec(str(code), globals_, locals_)
            rsp = {'globals': globals_, 'locals': locals_}
            if 'res' in locals_:
                rsp['locals_res'] = locals_['res']
            if 'res' in globals_:
                rsp['globals_res'] = globals_['res']
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

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = code
            metadata['output'] = rsp

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        pass
