from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
import traceback
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

from .backend import engines
from .backend.base import ENGINE_UNREGISTERED, Engine
from .context import CURRENT_ENGINE_VAR
from .prompts import (
    ProbabilisticBooleanModeMedium,
    ProbabilisticBooleanModeStrict,
    ProbabilisticBooleanModeTolerant,
)
from .utils import UserMessage

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from .core import Argument
    from .post_processors import PostProcessor
    from .pre_processors import PreProcessor
else:
    Callable = Any
    ModuleType = type(importlib)
    PostProcessor = PreProcessor = Any


class ConstraintViolationException(Exception):
    pass


class ProbabilisticBooleanMode(Enum):
    STRICT = 0
    MEDIUM = 1
    TOLERANT = 2


ENGINE_PROBABILISTIC_BOOLEAN_MODE = ProbabilisticBooleanMode.MEDIUM



def _probabilistic_bool(rsp: str, mode=ProbabilisticBooleanMode.TOLERANT) -> bool:
    if rsp is None:
        return False
    # check if rsp is a string / hard match
    val = str(rsp).lower()
    if   mode == ProbabilisticBooleanMode.STRICT:
        return val == ProbabilisticBooleanModeStrict
    if mode == ProbabilisticBooleanMode.MEDIUM:
        return val in ProbabilisticBooleanModeMedium
    if mode == ProbabilisticBooleanMode.TOLERANT:
        # allow for probabilistic boolean / fault tolerance
        return val in ProbabilisticBooleanModeTolerant
    UserMessage(f"Invalid mode {mode} for probabilistic boolean!", raise_with=ValueError)
    return False


def _cast_collection_response(rsp: Any, return_constraint: type) -> Any:
    try:
        res = ast.literal_eval(rsp)
    except Exception:
        logger.warning(f"Failed to cast return type to {return_constraint} for {rsp!s}")
        warnings.warn(f"Failed to cast return type to {return_constraint}", stacklevel=2)
        res = rsp
    assert res is not None, f"Return type cast failed! Check if the return type is correct or post_processors output matches desired format: {rsp!s}"
    return res


def _cast_boolean_response(rsp: Any, mode: ProbabilisticBooleanMode) -> bool:
    if len(rsp) <= 0:
        return False
    return _probabilistic_bool(rsp, mode=mode)


def _cast_with_fallback(rsp: Any, return_constraint: type) -> Any:
    try:
        return return_constraint(rsp)
    except (ValueError, TypeError):
        if return_constraint is int:
            UserMessage(f"Cannot convert {rsp} to int", raise_with=ConstraintViolationException)
        warnings.warn(f"Failed to cast {rsp} to {return_constraint}", stacklevel=2)
        return rsp


def _cast_return_type(rsp: Any, return_constraint: type, engine_probabilistic_boolean_mode: ProbabilisticBooleanMode) -> Any:
    if return_constraint is inspect._empty:
        return rsp
    if issubclass(return_constraint, BaseModel):
        # pydantic model
        return return_constraint(data=rsp)
    if str(return_constraint) == str(type(rsp)):
        return rsp
    if return_constraint in (list, tuple, set, dict):
        return _cast_collection_response(rsp, return_constraint)
    if return_constraint is bool:
        return _cast_boolean_response(rsp, mode=engine_probabilistic_boolean_mode)
    if not isinstance(rsp, return_constraint):
        return _cast_with_fallback(rsp, return_constraint)
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
            UserMessage(f"Constraint not satisfied for value {rsp!r} with constraint {constraint}", raise_with=ConstraintViolationException)
    return rsp, metadata


def _apply_preprocessors(argument, instance: Any, pre_processors: list[PreProcessor] | None) -> str:
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
        if return_type is str and isinstance(rsp, list):
            return '\n'.join(rsp[:limit_])
        if return_type is list:
            return rsp[:limit_]
        if return_type is dict:
            keys = list(rsp.keys())
            return {k: rsp[k] for k in keys[:limit_]}
        if return_type is set:
            return set(list(rsp)[:limit_])
        if return_type is tuple:
            return tuple(list(rsp)[:limit_])
    return rsp


def _prepare_argument(argument: Any, engine: Any, instance: Any, func: Callable, constraints: list[Callable], default: Any, limit: int, trials: int, pre_processors: list[PreProcessor] | None, post_processors: list[PostProcessor] | None) -> Any:
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
    try:
        rsp = func(
            instance,
            *argument.args,
            error=error,
            stack_trace=stack_trace,
            **argument.signature_kwargs,
        )
    except Exception:
        raise error from None  # Re-raise the original error without chaining fallback failure.
    if rsp is not None:
        # fallback was implemented
        return {"data": rsp, "error": error, "stack_trace": stack_trace}
    if argument.prop.default is not None:
        # no fallback implementation, but default value is set
        return {"data": argument.prop.default, "error": error, "stack_trace": stack_trace}
    raise error from None


def _process_query_single(engine,
                          instance,
                          func: Callable,
                          constraints: list[Callable] | None = None,
                          default: object | None = None,
                          limit: int = 1,
                          trials: int = 1,
                          pre_processors: list[PreProcessor] | None = None,
                          post_processors: list[PostProcessor] | None = None,
                          argument=None):
    if constraints is None:
        constraints = []
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
            logger.error(f"Failed to execute query: {e!s}")
            logger.error(f"Stack trace: {stack_trace}")
            if _ == trials - 1:
                result = _execute_query_fallback(func, instance, argument, error=e, stack_trace=stack_trace)
                if result is None:
                    raise e

    limited_result = _limit_number_results(result, argument, argument.prop.return_constraint)
    if argument.prop.return_metadata:
        return limited_result, metadata
    return limited_result


def _normalize_processors(pre_processors: list[PreProcessor] | PreProcessor | None,
                          post_processors: list[PostProcessor] | PostProcessor | None) -> tuple[list[PreProcessor] | None, list[PostProcessor] | None]:
    if pre_processors and not isinstance(pre_processors, list):
        pre_processors = [pre_processors]
    if post_processors and not isinstance(post_processors, list):
        post_processors = [post_processors]
    return pre_processors, post_processors


def _run_query_with_retries(
        engine: Engine,
        argument: Any,
        func: Callable,
        instance: Any,
        trials: int,
        return_constraint: type,
        post_processors: list[PostProcessor] | None,
    ) -> tuple[Any, Any]:
    try_cnt = 0
    rsp = None
    metadata = None
    while try_cnt < trials:
        try_cnt += 1
        try:
            outputs = _execute_query(engine, argument)
            rsp, metadata = _apply_postprocessors(outputs, return_constraint, post_processors, argument)
            break
        except Exception as error:
            stack_trace = traceback.format_exc()
            logger.error(f"Failed to execute query: {error!s}")
            logger.error(f"Stack trace: {stack_trace}")
            if try_cnt < trials:
                continue
            rsp = _execute_query_fallback(func, instance, argument, error=error, stack_trace=stack_trace)
            metadata = None
    return rsp, metadata


def _execute_query(engine, argument) -> list[object]:
    # build prompt and query engine
    engine.prepare(argument)
    if argument.prop.preview:
        return engine.preview(argument)
    return engine(argument) # currently only supports single query


def _process_query(
        engine: Engine,
        instance: Any,
        func: Callable,
        constraints: list[Callable] | None = None,
        default: object | None = None,
        limit: int | None = None,
        trials: int = 1,
        pre_processors: list[PreProcessor] | None = None,
        post_processors: list[PostProcessor] | None = None,
        argument: Argument = None,
    ) -> Any:

    if constraints is None:
        constraints = []
    pre_processors, post_processors = _normalize_processors(pre_processors, post_processors)

    argument = _prepare_argument(argument, engine, instance, func, constraints, default, limit, trials, pre_processors, post_processors)
    return_constraint = argument.prop.return_constraint
    # if prep_processors is empty or none this returns an empty string
    processed_input = _apply_preprocessors(argument, instance, pre_processors)
    if not argument.prop.raw_input:
        argument.prop.processed_input = processed_input

    rsp, metadata = _run_query_with_retries(engine, argument, func, instance, trials, return_constraint, post_processors)
    if argument.prop.preview:
        if argument.prop.return_metadata:
            return rsp, metadata
        return rsp

    if not argument.prop.raw_output:
        rsp = _limit_number_results(rsp, argument, return_constraint)
    if argument.prop.return_metadata:
        return rsp, metadata
    return rsp


class EngineRepository:
    _instance = None

    def __init__(self):
        if '_engines' not in self.__dict__:  # ensures _engines is only set once
            self._engines: dict[str, Engine] = {}

    def __new__(cls, *_args, **_kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()  # Explicitly call __init__
        return cls._instance

    @staticmethod
    def register(id: str, engine_instance: Engine, allow_engine_override: bool = False) -> None:
        self = EngineRepository()
        # Check if the engine is already registered
        if id in self._engines and not allow_engine_override:
            UserMessage(f"Engine {id} is already registered. Set allow_engine_override to True to override.", raise_with=ValueError)

        self._engines[id] = engine_instance

    @staticmethod
    def register_from_plugin(id: str, plugin: str, selected_engine: str | None = None, allow_engine_override: bool = False, *args, **kwargs) -> None:
        # Lazy import keeps functional -> imports -> symbol -> core -> functional cycle broken.
        from .imports import Import # noqa
        types = Import.load_module_class(plugin)
        # filter out engine class type
        engines = [t for t in types if issubclass(t, Engine) and t is not Engine]
        if len(engines) > 1 and selected_engine is None:
            UserMessage(f"Multiple engines found in plugin {plugin}. Please specify the engine to use.", raise_with=ValueError)
        if len(engines) > 1 and selected_engine is not None:
            engine = [e for e in engines if selected_engine in str(e)]
            if len(engine) <= 0:
                UserMessage(f"No engine named {selected_engine} found in plugin {plugin}.", raise_with=ValueError)
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
                            UserMessage(f"Engine {instance!s} does not have an id. Please add a method id() to the class.", raise_with=ValueError)
                        # call engine_() to get the id of the engine
                        id_ = engine_id_func_()
                        # only registered configured engine
                        if id_ != ENGINE_UNREGISTERED:
                            # register new engine
                            self.register(id_, instance, allow_engine_override=allow_engine_override)
                    except Exception as e:
                        logger.error(f"Failed to register engine {attribute!s}: {e!s}")

    @staticmethod
    def get(engine_name: str, *_args, **_kwargs):

        self = EngineRepository()
        # First check if we're in the context manager that dynamically changes models
        if engine_name == "neurosymbolic":
            engine = self.get_dynamic_engine_instance()
            if engine is not None:
                return engine

        # Otherwise, fallback to normal lookup:
        if engine_name not in self._engines:
            subpackage_name = engine_name.replace('-', '_')
            subpackage = importlib.import_module(f"{engines.__package__}.{subpackage_name}", None)
            if subpackage is None:
                UserMessage(f"The symbolicai library does not contain the engine named {engine_name}.", raise_with=ValueError)
            self.register_from_package(subpackage)
        engine = self._engines.get(engine_name, None)
        if engine is None:
            UserMessage(f"No engine named {engine_name} is registered.", raise_with=ValueError)
        return engine

    @staticmethod
    def list() -> list[str]:
        self = EngineRepository()
        return dict(self._engines.items())

    @staticmethod
    def command(engines: list[str], *args, **kwargs) -> Any:
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
        UserMessage(f"No engine named <{engine_name}> is registered.", raise_with=ValueError)
        return None

    @staticmethod
    def query(engine: str, *args, **kwargs) -> tuple:
        self = EngineRepository()
        engine = self.get(engine)
        if engine:
            engine_allows_batching = getattr(engine, 'allows_batching', False)
            if engine_allows_batching:
                return _process_query_single(engine, *args, **kwargs)
            return _process_query(engine, *args, **kwargs)
        UserMessage(f"No engine named {engine} is registered.", raise_with=ValueError)
        return None

    @staticmethod
    def bind_property(engine: str, property: str, *_args, **_kwargs):
        self = EngineRepository()
        """Bind a property to a specific engine."""
        engine = self.get(engine)
        if engine:
            return getattr(engine, property, None)
        UserMessage(f"No engine named {engine} is registered.", raise_with=ValueError)
        return None

    def get_dynamic_engine_instance(self):
        # 1) Primary: use ContextVar (fast, async-safe)
        try:
            eng = CURRENT_ENGINE_VAR.get()
            if eng is not None:
                return eng
        except Exception:
            pass

        # 2) Fallback: walk ONLY current thread frames (legacy behavior)
        # Keeping DynamicEngine import lazy prevents functional importing components before it finishes loading.
        from .components import DynamicEngine # noqa
        try:
            frame = sys._getframe()
        except Exception:
            return None
        while frame:
            try:
                locals_copy = frame.f_locals.copy() if hasattr(frame.f_locals, 'copy') else dict(frame.f_locals)
            except Exception:
                UserMessage(
                    "Unexpected failure copying frame locals while resolving DynamicEngine.",
                    raise_with=None,
                )
                locals_copy = {}
            for value in locals_copy.values():
                if isinstance(value, DynamicEngine) and getattr(value, '_entered', False):
                    return value.engine_instance
            frame = frame.f_back
        return None
