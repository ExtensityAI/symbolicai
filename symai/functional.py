from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
import traceback
import warnings
from enum import Enum
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from box import Box
from loguru import logger
from pydantic import BaseModel

from .backend import engines
from .backend.base import ENGINE_UNREGISTERED, Engine
from .post_processors import PostProcessor
from .pre_processors import PreProcessor


class ConstraintViolationException(Exception):
    pass


class ProbabilisticBooleanMode(Enum):
    STRICT = 0
    MEDIUM = 1
    TOLERANT = 2


ENGINE_PROBABILISTIC_BOOLEAN_MODE = ProbabilisticBooleanMode.MEDIUM
from .prompts import (ProbabilisticBooleanModeMedium,
                      ProbabilisticBooleanModeStrict,
                      ProbabilisticBooleanModeTolerant)


def _probabilistic_bool(rsp: str, mode=ProbabilisticBooleanMode.TOLERANT) -> bool:
    if rsp is None:
        return False
    # check if rsp is a string / hard match
    val = str(rsp).lower()
    if   mode == ProbabilisticBooleanMode.STRICT:
        return val == ProbabilisticBooleanModeStrict
    elif mode == ProbabilisticBooleanMode.MEDIUM:
        return val in ProbabilisticBooleanModeMedium
    elif mode == ProbabilisticBooleanMode.TOLERANT:
        # allow for probabilistic boolean / fault tolerance
        return val in ProbabilisticBooleanModeTolerant
    else:
        raise ValueError(f"Invalid mode {mode} for probabilistic boolean!")


def _cast_return_type(rsp: Any, return_constraint: Type, engine_probabilistic_boolean_mode: ProbabilisticBooleanMode) -> Any:
    if return_constraint == inspect._empty:
        # do not cast if return type is not specified
        pass
    elif issubclass(return_constraint, BaseModel):
        # pydantic model
        rsp = return_constraint(data=rsp)
        return rsp
    elif str(return_constraint) == str(type(rsp)):
        return rsp
    elif return_constraint in (list, tuple, set, dict):
        try:
            res = ast.literal_eval(rsp)
        except Exception as e:
            logger.warning(f"Failed to cast return type to {return_constraint} for {str(rsp)}")
            warnings.warn(f"Failed to cast return type to {return_constraint}")  # Add warning for test
            res = rsp
        assert res is not None, f"Return type cast failed! Check if the return type is correct or post_processors output matches desired format: {str(rsp)}"
        return res
    elif return_constraint == bool:
        if len(rsp) <= 0:
            return False
        else:
            return _probabilistic_bool(rsp, mode=engine_probabilistic_boolean_mode)
    elif not isinstance(rsp, return_constraint):
        try:
            # hard cast to return type fallback
            rsp = return_constraint(rsp)
        except (ValueError, TypeError) as e:
            if return_constraint == int:
                raise ConstraintViolationException(f"Cannot convert {rsp} to int")
            warnings.warn(f"Failed to cast {rsp} to {return_constraint}")
            return rsp
    return rsp

def _apply_postprocessors(outputs, return_constraint, post_processors, argument, mode=ENGINE_PROBABILISTIC_BOOLEAN_MODE):
    if argument.prop.preview:
        return outputs
    
    rsp, metadata = outputs[0][0], outputs[1]
    argument.prop.outputs = outputs
    argument.prop.metadata = metadata

    if argument.prop.raw_output:
        return metadata.get('raw_output'), metadata

    if post_processors:
        for pp in post_processors:
            rsp = pp(rsp, argument)
    rsp = _cast_return_type(rsp, return_constraint, mode)

    for constraint in argument.prop.constraints:
        if not constraint(rsp):
            raise ConstraintViolationException("Constraint not satisfied:", rsp, constraint)
    return rsp, metadata


def _apply_preprocessors(argument, instance: Any, pre_processors: Optional[List[PreProcessor]]) -> str:
    processed_input = ''
    if pre_processors and not argument.prop.raw_input:
        argument.prop.instance = instance
        for pp in pre_processors:
            t = pp(argument)
            processed_input += t if t is not None else ''
    else:
        if argument.args and len(argument.args) > 0:
            processed_input      += ' '.join([str(a) for a in argument.args])
    return processed_input


def _limit_number_results(rsp: Any, argument, return_type):
    limit_ = argument.prop.limit if argument.prop.limit else (len(rsp) if hasattr(rsp, '__len__') else None)
    # the following line is different from original code to make it work for iterable return types when the limit is 1
    if limit_ is not None:
        if return_type == str and isinstance(rsp, list):
            return '\n'.join(rsp[:limit_])
        elif return_type == list:
            return rsp[:limit_]
        elif return_type == dict:
            keys = list(rsp.keys())
            return {k: rsp[k] for k in keys[:limit_]}
        elif return_type == set:
            return set(list(rsp)[:limit_])
        elif return_type == tuple:
            return tuple(list(rsp)[:limit_])
    return rsp


def _prepare_argument(argument: Any, engine: Any, instance: Any, func: Callable, constraints: List[Callable], default: Any, limit: int, trials: int, pre_processors: Optional[List[PreProcessor]], post_processors: Optional[List[PostProcessor]]) -> Any:
    # check signature for return type
    sig = inspect.signature(func)
    return_constraint = sig._return_annotation
    assert 'typing' not in str(return_constraint), "Return type must be of base type not generic Typing object, e.g. int, str, list, etc."

    # prepare argument container
    argument.prop.engine            = engine
    argument.prop.instance          = instance
    argument.prop.instance_type     = type(instance)
    argument.prop.signature         = sig
    argument.prop.func              = func
    argument.prop.constraints       = constraints
    argument.prop.return_constraint = return_constraint
    argument.prop.default           = default
    argument.prop.limit             = limit
    argument.prop.trials            = trials
    argument.prop.pre_processors    = pre_processors
    argument.prop.post_processors   = post_processors
    return argument


def _execute_query_fallback(func, instance, argument, error=None, stack_trace=None):
    """Execute fallback behavior when query execution fails.
    
    This matches the fallback logic used in _process_query by handling errors consistently,
    providing error context to the fallback function, and maintaining the same return format.
    """
    rsp = func(instance, error=error, stack_trace=stack_trace, *argument.args, **argument.signature_kwargs)
    if rsp is not None:
        # fallback was implemented
        rsp = dict(data=rsp, error=error, stack_trace=stack_trace)
        return rsp
    elif argument.prop.default is not None:
        # no fallback implementation, but default value is set
        rsp = dict(data=argument.prop.default, error=error, stack_trace=stack_trace)
        return rsp
    else:
        raise error


def _process_query_single(engine,
                          instance,
                          func: Callable,
                          constraints: List[Callable] = [],
                          default: Optional[object] = None,
                          limit: int = 1,
                          trials: int = 1,
                          pre_processors: Optional[List[PreProcessor]] = None,
                          post_processors: Optional[List[PostProcessor]] = None,
                          argument=None):
    if pre_processors and not isinstance(pre_processors, list):
        pre_processors = [pre_processors]
    if post_processors and not isinstance(post_processors, list):
        post_processors = [post_processors]

    argument = _prepare_argument(argument, engine, instance, func, constraints, default, limit, trials, pre_processors, post_processors)

    preprocessed_input = _apply_preprocessors(argument, instance, pre_processors)
    argument.prop.processed_input = preprocessed_input
    engine.prepare(argument)

    result = None
    metadata = None
    for _ in range(trials):
        try:
            outputs = engine.executor_callback(argument)
            result, metadata = _apply_postprocessors(outputs, argument.prop.return_constraint, post_processors, argument)
            break
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Failed to execute query: {str(e)}")
            logger.error(f"Stack trace: {stack_trace}")
            if _ == trials - 1:
                result = _execute_query_fallback(func, instance, argument, error=e, stack_trace=stack_trace)
                if result is None:
                    raise e

    limited_result = _limit_number_results(result, argument, argument.prop.return_constraint)
    if argument.prop.return_metadata:
        return limited_result, metadata
    return limited_result


def _execute_query(engine, argument) -> List[object]:
    # build prompt and query engine
    engine.prepare(argument)
    if argument.prop.preview:
        return engine.preview(argument)
    outputs = engine(argument) # currently only supports single query
    return outputs


def _process_query(
        engine: Engine,
        instance: Any,
        func: Callable,
        constraints: List[Callable] = [],
        default: Optional[object] = None,
        limit: int = 1,
        trials: int = 1,
        pre_processors: Optional[List[PreProcessor]] = None,
        post_processors: Optional[List[PostProcessor]] = None,
        argument: Argument = None,
    ) -> Any:

    if pre_processors and not isinstance(pre_processors, list):
        pre_processors = [pre_processors]
    if post_processors and not isinstance(post_processors, list):
        post_processors = [post_processors]

    argument = _prepare_argument(argument, engine, instance, func, constraints, default, limit, trials, pre_processors, post_processors)
    return_constraint = argument.prop.return_constraint
    # if prep_processors is empty or none this returns an empty string 
    processed_input = _apply_preprocessors(argument, instance, pre_processors)
    if not argument.prop.raw_input:
        argument.prop.processed_input = processed_input

    try_cnt = 0
    while try_cnt < trials:
        try_cnt += 1
        try:
            outputs = _execute_query(engine, argument)
            rsp, metadata = _apply_postprocessors(outputs, return_constraint, post_processors, argument)
            if argument.prop.preview:
                if argument.prop.return_metadata:
                    return rsp, metadata
                return rsp

        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Failed to execute query: {str(e)}")
            logger.error(f"Stack trace: {stack_trace}")
            if try_cnt < trials:
                continue      
            rsp = _execute_query_fallback(func, instance, argument, error=e, stack_trace=stack_trace)

    rsp = _limit_number_results(rsp, argument, return_constraint)
    if argument.prop.return_metadata:
        return rsp, metadata
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
        return cls._instance

    @staticmethod
    def register(id: str, engine_instance: Engine, allow_engine_override: bool = False) -> None:
        self = EngineRepository()
        # Check if the engine is already registered
        if id in self._engines and not allow_engine_override:
            raise ValueError(f"Engine {id} is already registered. Set allow_engine_override to True to override.")

        self._engines[id] = engine_instance

    @staticmethod
    def register_from_plugin(id: str, plugin: str, selected_engine: Optional[str] = None, allow_engine_override: bool = False, *args, **kwargs) -> None:
        from .imports import Import
        types = Import.load_module_class(plugin)
        # filter out engine class type
        engines = [t for t in types if issubclass(t, Engine) and t is not Engine]
        if len(engines) > 1 and selected_engine is None:
            raise ValueError(f"Multiple engines found in plugin {plugin}. Please specify the engine to use.")
        elif len(engines) > 1 and selected_engine is not None:
            engine = [e for e in engines if selected_engine in str(e)]
            if len(engine) <= 0:
                raise ValueError(f"No engine named {selected_engine} found in plugin {plugin}.")
        engine = engines[0](*args, **kwargs)
        EngineRepository.register(id, engine, allow_engine_override=allow_engine_override)

    @staticmethod
    def register_from_package(package: ModuleType, allow_engine_override: bool = False, *args, **kwargs) -> None:
        self = EngineRepository()
        # Iterate over all modules in the given package and import them
        for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            module = importlib.import_module(module_name)

            # Check all classes defined in the module
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)

                # Register class if it is a subclass of Engine (but not Engine itself)
                if inspect.isclass(attribute) and issubclass(attribute, Engine) and attribute is not Engine:
                    try:
                        instance = attribute(*args, **kwargs) # Create an instance of the engine class
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
                    except Exception as e:
                        logger.error(f"Failed to register engine {str(attribute)}: {str(e)}")

    @staticmethod
    def get(engine_name: str, *args, **kwargs):

        self = EngineRepository()
        # First check if we're in the context manager that dynamically changes models
        if engine_name == "neurosymbolic":
            engine = self.get_dynamic_engine_instance()
            if engine is not None:
                return engine

        # Otherwise, fallback to normal lookup:
        if engine_name not in self._engines.keys():
            subpackage_name = engine_name.replace('-', '_')
            subpackage = importlib.import_module(f"{engines.__package__}.{subpackage_name}", None)
            if subpackage is None:
                raise ValueError(f"The symbolicai library does not contain the engine named {engine_name}.")
            self.register_from_package(subpackage)
        engine = self._engines.get(engine_name, None)
        if engine is None:
            raise ValueError(f"No engine named {engine_name} is registered.")
        return engine

    @staticmethod
    def list() -> List[str]:
        self = EngineRepository()
        return dict(self._engines.items())

    @staticmethod
    def command(engines: List[str], *args, **kwargs) -> Any:
        self = EngineRepository()
        if isinstance(engines, str):
            engines = [engines]
        if 'all' in engines:
            # Call the command function for all registered engines with provided arguments
            return [engine.command(*args, **kwargs) for name, engine in self._engines.items()]
        # Call the command function for the engine with provided arguments
        for engine_name in engines:
            engine = self.get(engine_name)
            if engine:
                # Call the command function for the engine with provided arguments
                return engine.command(*args, **kwargs)
        raise ValueError(f"No engine named <{engine_name}> is registered.")

    @staticmethod
    def query(engine: str, *args, **kwargs) -> Tuple:
        self = EngineRepository()
        engine = self.get(engine)
        if engine:
            engine_allows_batching = getattr(engine, 'allows_batching', False)
            if engine_allows_batching:
                return _process_query_single(engine, *args, **kwargs)
            else:
                return _process_query(engine, *args, **kwargs)
        raise ValueError(f"No engine named {engine} is registered.")

    @staticmethod
    def bind_property(engine: str, property: str, *args, **kwargs):
        self = EngineRepository()
        """Bind a property to a specific engine."""
        engine = self.get(engine)
        if engine:
            return getattr(engine, property, None)
        raise ValueError(f"No engine named {engine} is registered.")

    def get_dynamic_engine_instance(self):
        from .components import DynamicEngine

        # Iterate over all thread frames
        for thread_id, top_frame in sys._current_frames().items():
            frame = top_frame
            while frame:
                locals_copy = frame.f_locals.copy()
                for _, value in locals_copy.items():
                    if isinstance(value, DynamicEngine) and getattr(value, '_entered', False):
                        return value.engine_instance
                frame = frame.f_back
        return None
