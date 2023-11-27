import ast
import inspect
import os
import pickle
import random
import time
import traceback
import importlib
import pkgutil

from pathlib import Path
from typing import Callable, Dict, List, Optional
from types import ModuleType
from typing import Dict, Any, Tuple

from .post_processors import *
from .pre_processors import *
from .prompts import Prompt
from .misc.console import ConsoleStyle
from .backend.base import Engine, ENGINE_UNREGISTERED
from .backend import engines


class ConstraintViolationException(Exception):
    pass


def _execute_query(engine, post_processors, wrp_self, wrp_params, return_constraint, args, kwargs) -> List[object]:
    # build prompt and query engine
    engine.prepare(args, kwargs, wrp_params)

    # return preview of the command if preview is set
    if 'preview' in wrp_params and wrp_params['preview']:
        return engine.preview(wrp_params)

    outputs  = engine(**wrp_params) # currently only support single query
    rsp      = outputs[0][0]
    metadata = outputs[1]

    if post_processors:
        for pp in post_processors:
            rsp = pp(wrp_self, wrp_params, rsp, *args, **kwargs)

    # check if return type cast
    if return_constraint == type(rsp):
        pass
    elif return_constraint == list or \
        return_constraint == tuple or \
            return_constraint == set or \
                return_constraint == dict:
        try:
            res = ast.literal_eval(rsp)
        except Exception as e:
            print('functional parsing failed', e, rsp)
            res = rsp
        assert res is not None, "Return type cast failed! Check if the return type is correct or post_processors output matches desired format: " + str(rsp)
        rsp = res
    elif return_constraint == bool:
        # do not cast with bool -> always returns true
        if len(rsp) <= 0:
            rsp = False
        else:
            rsp = str(rsp).lower() in "'true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'ok', ['true']"
    elif return_constraint == inspect._empty:
        pass
    else:
        rsp = return_constraint(rsp)

    # check if satisfies constraints
    for constraint in wrp_params['constraints']:
        if not constraint(rsp):
            raise ConstraintViolationException("Constraint not satisfied:", rsp, constraint)

    return rsp, metadata


def _process_query(engine,
                   instance,
                   func:                Callable,
                   prompt:              str,
                   examples:            Optional[Prompt]                = None,
                   constraints:         List[Callable]                  = [],
                   default:             Optional[object]                = None,
                   limit:               int                             = 1,
                   trials:              int                             = 1,
                   pre_processors:      Optional[List[PreProcessor]]    = None,
                   post_processors:     Optional[List[PostProcessor]]   = None,
                   *args, **kwargs):

    if pre_processors and not isinstance(pre_processors, list):
        pre_processors = [pre_processors]
    if post_processors and not isinstance(post_processors, list):
        post_processors = [post_processors]

    # check signature for return type
    sig = inspect.signature(func)
    return_constraint = sig._return_annotation
    assert 'typing' not in str(return_constraint), "Return type must be of base type not generic Typing object, e.g. int, str, list, etc."

    # prepare wrapper parameters
    wrp_params = {
        'wrp_self': instance,
        'func': func,
        'prompt': prompt,
        'examples': examples,
        'constraints': constraints,
        'default': default,
        'limit': limit,
        'signature': sig,
        **kwargs
    }

    # pre-process text
    suffix = ''
    if pre_processors and 'raw_input' not in wrp_params:
        for pp in pre_processors:
            t = pp(instance, wrp_params, *args, **kwargs)
            suffix += t if t is not None else ''
    else:
        if args and len(args) > 0:
            suffix += ' '.join([str(a) for a in args])
            suffix += '\n'
        if kwargs and len(kwargs) > 0:
            suffix += ' '.join([f'{k}: {v}' for k, v in kwargs.items()])
            suffix += '\n'
    wrp_params['processed_input'] = suffix

    # try run the function
    try_cnt  = 0
    while try_cnt < trials:
        try_cnt += 1
        try:
            rsp, metadata = _execute_query(engine, post_processors, instance, wrp_params, return_constraint, args, kwargs)
            # return preview of the command if preview is set
            if 'preview' in wrp_params and wrp_params['preview']:
                return rsp
        except Exception as e:
            print(f'ERROR: {str(e)}')
            traceback.print_exc()
            if try_cnt < trials:
                continue # repeat if query unsuccessful
            # if max retries reached, return default or raise exception
            # execute default function implementation as fallback
            f_kwargs = {}
            f_sig_params = list(sig.parameters)
            # handle self and kwargs to match function signature
            if  f_sig_params[0] == 'self':
                f_sig_params = f_sig_params[1:]
            # allow for args to be passed in as kwargs
            if len(kwargs) == 0 and len(args) > 0:
                for i, arg in enumerate(args):
                    f_kwargs[f_sig_params[i]] = arg
            # execute function or method based on self presence
            rsp = func(instance, *args, **kwargs)
            # if there is also no default implementation, raise exception
            if rsp is None and wrp_params['default'] is None:
                raise e # raise exception if no default and no function implementation
            elif rsp is None: # return default if there is one
                rsp = wrp_params['default']

    # return based on return type
    try:
        limit_ = wrp_params['limit'] if wrp_params['limit'] is not None else len(rsp)
    except:
        limit_ = None

    # if limit_ is greater than 1 and expected only single string return type, join the list into a string
    if limit_ is not None and limit_ > 1 and return_constraint == str and type(rsp) == list:
        rsp = '\n'.join(rsp[:limit_])
    elif limit_ is not None and limit_ > 1 and return_constraint == list:
        rsp = rsp[:limit_]
    elif limit_ is not None and limit_ > 1 and return_constraint == dict:
        keys = list(rsp.keys())
        rsp = {k: rsp[k] for k in keys[:limit_]}
    elif limit_ is not None and limit_ > 1 and return_constraint == set:
        rsp = set(list(rsp)[:limit_])
    elif limit_ is not None and limit_ > 1 and return_constraint == tuple:
        rsp = tuple(list(rsp)[:limit_])

    return rsp


def retry_func(
    func: callable,
    exceptions: Exception,
    tries: int,
    delay: int,
    max_delay: int,
    backoff=int,
    jitter=int
):
    _tries, _delay = tries, delay
    while _tries:
        try:
            return func()
        except exceptions as e:
            _tries -= 1
            if not _tries:
                raise

            time.sleep(_delay)
            _delay *= backoff

            if isinstance(jitter, tuple):
                _delay += random.uniform(*jitter)
            else:
                _delay += jitter

            if max_delay >= 0:
                _delay = min(_delay, max_delay)


def cache_registry_func(
        in_memory: bool,
        cache_path: str,
        func: Callable,
        *args, **kwargs
    ):

    if not os.path.exists(cache_path): os.makedirs(cache_path)

    if in_memory and os.path.exists(Path(cache_path) / func.__qualname__):
        with open(Path(cache_path) / func.__qualname__, 'rb') as f:
            call = pickle.load(f)

        return call

    call = func(*args, **kwargs)
    with open(Path(cache_path) / func.__qualname__, 'wb') as f:
        pickle.dump(call , f)

    return call


class EngineRepository(object):
    _instance = None

    def __init__(self):
        self._engines: Dict[str, Engine] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EngineRepository, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def register(cls, id: str, engine_instance: Engine, allow_engine_override: bool = False) -> None:
        self = cls()
        # Check if the engine is already registered
        if id in self._engines.keys() and not allow_engine_override:
            raise ValueError(f"Engine {id} is already registered. Set allow_engine_override to True to override.")
        elif id in self._engines and allow_engine_override:
            reg_eng = self.get(id)
            with ConsoleStyle('warn', logging=True) as console:
                console.print(f"Engine {id} is already registered. Overriding engine: {reg_eng.__name__} with {str(str(engine_instance))}")
        else:
            with ConsoleStyle('debug', logging=False) as console:
                console.print(f"Registering engine: {id} >> {str(engine_instance)}")
        # Create an instance of the engine class and store it
        self._engines[id] = engine_instance

    @classmethod
    def register_from_package(cls, package: ModuleType, allow_engine_override: bool = False, *args, **kwargs) -> None:
        # Iterate over all modules in the given package and import them
        for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            module = importlib.import_module(module_name)

            # Check all classes defined in the module
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)

                # Register class if it is a subclass of Engine (but not Engine itself)
                if inspect.isclass(attribute) and issubclass(attribute, Engine) and attribute is not Engine:
                    instance = attribute() # Create an instance of the engine class
                    # Assume the class has an 'init' static method to initialize it
                    engine_id_func_ = getattr(instance, 'id', None)
                    if engine_id_func_ is None:
                        raise ValueError(f"Engine {str(instance)} does not have an id. Please add a method id() to the class.")
                    # call engine_() to get the id of the engine
                    id_ = engine_id_func_()
                    # only registered configured engine
                    if id_ != ENGINE_UNREGISTERED:
                        # register new engine
                        cls.register(id_, instance, allow_engine_override=allow_engine_override)

    @classmethod
    def get(cls, engine_name: str, *args, **kwargs) -> Engine:
        self = cls()
        if engine_name not in self._engines:
            # initialize engine
            raise ValueError(f"No engine named {engine_name} is registered.")
        return self._engines.get(engine_name)

    @classmethod
    def execute_command(cls, engine_name: str, *args, **kwargs) -> Any:
        self = cls()
        if engine_name == 'all':
            # Call the command function for all registered engines with provided arguments
            return [engine.command(engine, *args, **kwargs) for engine_name, engine in self._engines.items()]
        engine = cls.get(engine_name)
        if engine:
            # Call the command function for the engine with provided arguments
            return engine.command(engine, *args, **kwargs)
        raise ValueError(f"No engine named <{engine_name}> is registered.")

    @classmethod
    def process_query(cls, engine: str, *args, **kwargs) -> Tuple:
        engine = cls.get(engine)
        if engine:
            return _process_query(*args, **kwargs)
        raise ValueError(f"No engine named {engine} is registered.")

    @classmethod
    def bind_property(cls, engine_name: str, property_name: str, *args, **kwargs):
        """Bind a property to a specific engine."""
        engine = cls.get(engine_name)
        if engine:
            return getattr(engine, property_name, None)
        raise ValueError(f"No engine named {engine_name} is registered.")


EngineRepository.register_from_package(engines)
