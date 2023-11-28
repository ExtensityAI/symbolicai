import ast
import inspect
import traceback
import importlib
import pkgutil
import logging

from typing import Callable, Dict, List, Optional, Union
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
            logging.warn(f"Failed to cast return type to {return_constraint} for {str(rsp)}")
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
            logging.error(f"Failed to execute query: {str(e)}")
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


class EngineRepository(object):
    _instance = None

    def __init__(self):
        if '_engines' not in self.__dict__:  # ensures _engines is only set once
            self._engines: Dict[str, Engine] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EngineRepository, cls).__new__(cls, *args, **kwargs)
            cls._instance.__init__()  # Explicitly call __init__
            cls._instance.register_from_package(engines)
        return cls._instance

    def register(self, id: str, engine_instance: Engine, allow_engine_override: bool = False) -> None:
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

    def register_from_package(self, package: ModuleType, allow_engine_override: bool = False, *args, **kwargs) -> None:
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
                        self.register(id_, instance, allow_engine_override=allow_engine_override)

    def get(self, engine_name: str, *args, **kwargs) -> Engine:
        if engine_name not in self._engines.keys():
            # initialize engine
            raise ValueError(f"No engine named {engine_name} is registered.")
        return self._engines.get(engine_name)

    def list(self, *args, **kwargs) -> List[str]:
        return list(self._engines.keys())

    def execute_command(self, engines: Union[str, List[str]], *args, **kwargs) -> Any:
        if isinstance(engines, str):
            engines = [engines]
        if 'all' in engines:
            # Call the command function for all registered engines with provided arguments
            return [engine.command(engine, *args, **kwargs) for name, engine in self._engines.items()]
        # Call the command function for the engine with provided arguments
        for engine_name in engines:
            engine = self.get(engine_name)
            if engine:
                # Call the command function for the engine with provided arguments
                return engine.command(engine, *args, **kwargs)
        raise ValueError(f"No engine named <{engine_name}> is registered.")

    def process_query(self, engine: str, *args, **kwargs) -> Tuple:
        engine = self.get(engine)
        if engine:
            return _process_query(engine, *args, **kwargs)
        raise ValueError(f"No engine named {engine} is registered.")

    def bind_property(self, engine_name: str, property_name: str, *args, **kwargs):
        """Bind a property to a specific engine."""
        engine = self.get(engine_name)
        if engine:
            return getattr(engine, property_name, None)
        raise ValueError(f"No engine named {engine_name} is registered.")
