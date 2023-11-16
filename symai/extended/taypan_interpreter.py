from ..core import *
from ..pre_processors import PreProcessor
from ..symbol import Expression, Symbol
import pathlib


package_path = pathlib.Path(__file__).parent.parent.absolute()


with open(package_path / 'symbol.py', 'r') as f:
    SYMBOL_API = f.read()


with open(package_path / 'components.py', 'r') as f:
    COMPONENTS_API = f.read()


TAYPAN_DESCRIPTION = """[Description]
You are a programming language re-writing system from Taypan (high-level general-purpose programming language based on neuro-symbolic virtual machine) to Python interpreter, analogous to the relation between Scala and Java is the relation of Taypan to Python.

All code from Python is valid Taypan code, but not all code from Taypan is valid Python code.
Code from Taypan is translated to Python code and then executed by the Python interpreter.

The Taypan language extends concepts from Python by adding new features, and utilizing a neuro-symbolic framework `SymbolicAI` which is available with the `symai` import.

The `Symbol` class is the base class for all symbolic objects in `SymbolicAI` framework and has symbolic operations defined on it. When mappings from Taypan to Python are ambiguous, `Function` class or various other function from symbol can be used to define the mapping.

ALL REGULAR PYTHON CODE REMAINS UNCHANGED AND IS TRANSLATED TO ITSELF. ONLY TAYPAN CODE IS TRANSLATED TO PYTHON CODE.

The following new features added by Taypan:

- `par` keyword to run code in parallel (can be applied to any function or signature parameter):
```taypan
def concat_string(par x, y):
    return x + y

list_ = ['cat', 'dog', 'bird']
list_ = concat_string(list_)
print(list_) # Output: ['cat is an animal', 'dog is an animal', 'bird is an animal']
```

```python
def concat_string(x, y):
    from symai import parallel
    @parallel()
    def _concat_string_worker(x, y):
        return x + y
    import itertools # if needed
    return _concat_string_worker(x, itertools.repeat(y, len(x)))

list_ = ['cat', 'dog', 'bird']
list_ = concat_string(list_, ' is an animal')
print(list_) # Output: ['cat is an animal', 'dog is an animal', 'bird is an animal']
```

- `prot` keyword to define a protocol (interface) functions without implementation:
```taypan
prot my_protocol(*args, **kwargs):
    '''This is the function description in plain English
    '''
```

```python
def my_protocol(*args, **kwargs):
    from symai import Function, CodeExtractPostProcessor
    _func = Function('''[PYTHON_RETURN_VALUE]: This is the function description in plain English
    ''', post_processors=[CodeExtractPostProcessor()])
    return _func(*args, **kwargs)
```

- `sim` keyword to simulate a function call without actually calling it:
```taypan
def sim func1(data, *args, **kwargs) -> str:
    pass
```

```python
def func1(*args, **kwargs) -> str:
    from symai import Symbol
    sym = Symbol(data)
    return sym.simulate(*args, **kwargs)
```

- `api` keyword to define a dynamic API call function which is generated once at runtime and can be used to call any API:
```taypan
api func1(*args, **kwargs) -> str: # text of user request: ``
    '''This is the function description in plain English
    '''
```

```python
# all code must be self-contained in one function called `run` including all imports
def func1(*args, **kwargs) -> str: # [MANAGED] entry point cannot be changed
    from symai.extended import APIExecutor
    executor = APIExecutor()
    return executor('''This is the function description in plain English
    ''', *args, **kwargs)
```

[SymboliAI API]

- components `from symai.components import *`:
{0}

- symbol `from symai.symbol import *`:
{1}

""".format(COMPONENTS_API, SYMBOL_API)


class TaypanPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args, **kwds):
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return '```taypan\n{}\n =>'.format(str(args[0]))


class TaypanInterpreter(Expression):
    @property
    def static_context(self) -> str:
        return TAYPAN_DESCRIPTION

    def __init__(self):
        super().__init__()
        self.sym_return_type = TaypanInterpreter

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        @zero_shot(prompt="Translate the Taypan code to Python code:\n",
                   pre_processors=[TaypanPreProcessor()],
                   post_processors=[CodeExtractPostProcessor()], **kwargs)
        def _func(_, text) -> str:
            pass
        return _func(self, sym)
