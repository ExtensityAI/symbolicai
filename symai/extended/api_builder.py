from ..core import *
from ..formatter import SentenceFormatter
from ..post_processors import StripPostProcessor
from ..pre_processors import PreProcessor
from ..prompts import Prompt
from ..symbol import Expression, Symbol
from ..components import Execute, Try


API_BUILDER_DESCRIPTION = """[Description]
You are an API coding tool for Python that creates API calls to any web URL based on user requests.
For example, if the user wants to use the X API (former Twitter) to post a tweet, you will create the required API post call for that, i.e. 'Write Twitter post `hey, what's up` API-Key:...'.
If the user wants to use the X API to get the latest tweets, you will create the API call for that, e.g. 'Read Twitter post https://twitter.com/...'.
Each created function is atomic and can be used as a building block for more complex functions.
You can also create a function that calls other functions. However, all code must be self-contained in one function `run` including all imports.
Another constraint is that there is one mandatory function called `run` as an entry point to the executable runnable and one provided pre-build function that uses an large language model to extract and parse API calls parameters of user requests or manipulates string-based data as you see fit.
All code parts marked with [MANAGED] are strictly forbidden to be changed! They must be provided as is.
Always generate the entire code for the `run` function, including the `try` and `except` blocks, imports, etc. and the unchanged managed code parts.

For example, you can write yourself prompts to extract parameters from user requests and use them to create API calls:
```python
# all code must be self-contained in one function called `run` including all imports
def run(text: str) -> str: # [MANAGED] entry point cannot be changed
    # [MANAGED-BEGIN] mandatory imports here
    import traceback
    import requests
    from symai import Function
    # [MANAGED-END] mandatory imports here

    # optional imports here
    # TODO: all your imports and code here

    # executable code here
    try: # [MANAGED] must contain this line, do not change
        # optional helper functions here

        # optional params extraction here
        # TODO: extract params from request full-text if needed
        # Example:
        func = Function('YOUR_PROMPT_1') # TODO: extract function param 1
        param1 = func(request)
        func = Function('YOUR_PROMPT_2') # TODO: extract function param 2
        param2 = func(request)
        # ... extract more params if needed

        # optional params manipulation here
        res = # TODO: run https APIs with the respective params, use tools like requests, urllib, etc.

        # optional result formatting here
        # Another example:
        func = Function('YOUR_PROMPT_3') # TODO: format result if needed
        res = func(res)

        # mandatory return statement here
        res = str(res) # [MANAGED] must contain this line, do not change
        return res # [MANAGED] must return a string, do not change
    except Exception as e: # [MANAGED] must catch all exceptions and return them as string
        tb = traceback.format_exc() # [MANAGED] return full error stack trace as string
        return tb # [MANAGED] return tb as string, do not change

# mandatory statement here
res = run(value) # [MANAGED] must contain this line, do not change
```
"""


class APIBuilderPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args, **kwds):
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return '$> {} =>'.format(str(args[0]))


class APIBuilder(Expression):
    @property
    def static_context(self) -> str:
        return API_BUILDER_DESCRIPTION

    def __init__(self):
        super().__init__()
        self.sym_return_type = APIBuilder

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        @zero_shot(prompt="Build the API call code:\n",
                   pre_processors=[APIBuilderPreProcessor()],
                   post_processors=[CodeExtractPostProcessor()], **kwargs)
        def _func(_, text) -> str:
            pass

        return _func(self, sym)


class StackTraceRetryExecutor(Expression):
    def __init__(self, retries: int = 1):
        super().__init__()
        self.executor = Execute()
        self.max_retries = retries
        self._runnable = None

    def forward(self, code: Symbol, request: Symbol, **kwargs) -> Symbol:
        code = str(code)
        # Set value that gets passed on to the 'run' function in the generated code
        value = request.value # do not remove this line
        # Create the 'run' function
        self._runnable = self.executor(code, locals=locals().copy(), globals=globals().copy())
        result = self._runnable['locals']['run'](value)
        retry = 0
        # Retry if there is a 'Traceback' in the result
        while 'Traceback' in result and retry < self.max_retries:
            self._runnable = self.executor(code, payload=result, locals=locals().copy(), globals=globals().copy(), **kwargs)
            result = self._runnable['locals']['run'](value)
            retry += 1
        if 'locals_res' in self._runnable:
            result = self._runnable['locals_res']
        return result


class APIExecutor(Expression):
    def __init__(self, verbose=False, retries=1):
        super().__init__()
        self.builder = APIBuilder()
        self.executor = StackTraceRetryExecutor(retries=retries)

        self._verbose = verbose
        self._request = None
        self._code = None
        self._result = None

    @property
    def _runnable(self):
        return self.executor._runnable

    def forward(self, request: Symbol, **kwargs) -> Symbol:
        self._request = self._to_symbol(request)
        if self._verbose: print('[REQUEST]', self._request)
        # Generate the code to implement the API call
        self._code = self.builder(self._request)
        if self._verbose: print('[GENERATED_CODE]', self._code)
        # Execute the code to define the 'run' function
        self._result = self.executor(self._code, request=self._request)
        if self._verbose: print('[RESULT]:', self._result)
        self.value = self._result
        return self
