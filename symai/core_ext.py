import asyncio
import atexit
import functools
import logging
import multiprocessing as mp
import pickle
import random
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import dill
from loguru import logger

from . import __root_dir__
from .functional import EngineRepository
from .utils import UserMessage

logging.getLogger("multiprocessing").setLevel(logging.ERROR)

# -----------------------------------------------------------
# Global registry so we create **one** pool and reuse it
# -----------------------------------------------------------
_pool_registry: dict[int, mp.pool.Pool] = {}
_pool_lock = threading.Lock()

def _get_pool(workers: int) -> mp.pool.Pool:
    pool = _pool_registry.get(workers)
    if pool is not None:
        return pool
    with _pool_lock:
        pool = _pool_registry.get(workers)
        if pool is None:
            pool = mp.Pool(processes=workers)
            _pool_registry[workers] = pool
    return pool

@atexit.register
def _shutdown_pools() -> None:
    for pool in _pool_registry.values():
        pool.close()
        pool.join()

def _run_in_process(expr, func, args, kwargs):
    expr = dill.loads(expr)
    func = dill.loads(func)
    return func(expr, *args, **kwargs)

def _parallel(func: Callable, expressions: list[Callable], worker: int = mp.cpu_count() // 2):
    pickled_exprs = [dill.dumps(expr) for expr in expressions]
    pickled_func  = dill.dumps(func)
    pool = _get_pool(worker)
    def proxy_function(*args, **kwargs):
        return pool.starmap(
                _run_in_process,
                [(expr, pickled_func, args, kwargs) for expr in pickled_exprs]
            )
    return proxy_function

# Decorator
def parallel(expressions: list[Callable], worker: int = mp.cpu_count() // 2):
    def decorator_parallel(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Run expressions in parallel
            parallel_func = _parallel(func, expressions, worker=worker)
            # Call the proxy function to execute in parallel and capture results
            return parallel_func(*args, **kwargs)
        return wrapper
    return decorator_parallel

def bind(engine: str, property: str):
    '''
    Bind to an engine and retrieve one of its properties.
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*_args, **_kwargs):
            return EngineRepository.bind_property(
                    engine=engine,
                    property=property
                )
        return wrapper
    return decorator


def retry(
    exceptions=Exception,
    tries=-1,
    delay=0,
    max_delay=-1,
    backoff=1,
    jitter=0,
    graceful=False
):
    '''
    Returns a retry decorator for both async and sync functions.

    :param exceptions: an exception or a tuple of exceptions to catch. default: Exception.
    :param tries:      the maximum number of attempts. default: -1 (infinite).
    :param delay:      initial delay between attempts. default: 0.
    :param max_delay:  the maximum value of delay. default: -1 (no limit).
    :param backoff:    multiplier applied to delay between attempts. default: 1 (no backoff).
    :param jitter:     extra seconds added to delay between attempts. default: 0.
                       fixed if a number, random if a range tuple (min, max)
    :param graceful:   whether to raise an exception if all attempts fail or return with None. default: False.

    Credits to invlpg (https://pypi.org/project/retry)
    '''
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _aretry_func(
                    functools.partial(func, *args, **kwargs),
                    exceptions=exceptions,
                    tries=tries,
                    delay=delay,
                    max_delay=max_delay,
                    backoff=backoff,
                    jitter=jitter,
                    graceful=graceful
                )
            return async_wrapper
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _retry_func(
                functools.partial(func, *args, **kwargs),
                exceptions=exceptions,
                tries=tries,
                delay=delay,
                max_delay=max_delay,
                backoff=backoff,
                jitter=jitter,
                graceful=graceful
            )
        return sync_wrapper
    return decorator


def _retry_func(
    func: callable,
    exceptions: Exception,
    tries: int,
    delay: int,
    max_delay: int,
    backoff: int,
    jitter: int,
    graceful: bool
):
    _tries, _delay = tries, delay
    while _tries:
        try:
            return func()
        except exceptions:
            _tries -= 1
            if not _tries:
                if graceful:
                    return None
                raise

            time.sleep(_delay)
            _delay *= backoff

            if isinstance(jitter, tuple):
                _delay += random.uniform(*jitter)
            else:
                _delay += jitter

            if max_delay >= 0:
                _delay = min(_delay, max_delay)
    return None


async def _aretry_func(
    func: callable,
    exceptions: Exception,
    tries: int,
    delay: int,
    max_delay: int,
    backoff: int,
    jitter: int,
    graceful: bool
):
    _tries, _delay = tries, delay
    while _tries:
        try:
            return await func()
        except exceptions:
            _tries -= 1
            if not _tries:
                if graceful:
                    return None
                raise

            await asyncio.sleep(_delay)
            _delay *= backoff

            if isinstance(jitter, tuple):
                _delay += random.uniform(*jitter)
            else:
                _delay += jitter

            if max_delay >= 0:
                _delay = min(_delay, max_delay)
    return None


def cache(
    in_memory:  bool,
    cache_path: str = __root_dir__ / 'cache'
):
    '''
    Cache the result of a *any* function call. This is very useful in cost optimization (e.g. computing embeddings).
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance):
            return _cache_registry_func(
                    in_memory,
                    cache_path,
                    func,
                    instance
                )
        return wrapper
    return decorator


def _cache_registry_func(
        in_memory: bool,
        cache_path: str,
        func: Callable,
        *args, **kwargs
    ):

    cache_dir = Path(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / func.__qualname__

    if in_memory and cache_file.exists():
        with cache_file.open('rb') as f:
            return pickle.load(f)

    call = func(*args, **kwargs)
    with cache_file.open('wb') as f:
        pickle.dump(call , f)

    return call


def error_logging(debug: bool = False):
    '''
    Log the error of a function call.
    '''
    def dec(func):
        @functools.wraps(func)
        def _dec(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(e)
                if debug:
                    # Simple message:
                    UserMessage(f'Function: {func.__name__} call failed. Error: {e}')
                    # print traceback
                    traceback.print_exc()
                raise e
        return _dec
    return dec


def deprecated(reason: str = ""):
    """
    Mark a class, method, or function as deprecated.

    Args:
        reason (str): Explanation of why the item is deprecated and/or what to use instead

    Example usage:
        @deprecated("Use new_function() instead")
        def old_function(): pass

        @deprecated()
        class OldClass: pass

        class MyClass:
            @deprecated("Use new_method() instead")
            def old_method(self): pass
    """
    def decorator(obj):
        if isinstance(obj, type):
            # If obj is a class
            original_init = obj.__init__
            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                logger.warning(
                    f"{obj.__name__} is deprecated and will be removed in future versions. {reason}",
                    category=DeprecationWarning,
                    stacklevel=2
                )
                original_init(self, *args, **kwargs)
            obj.__init__ = new_init
            return obj

        # If obj is a function or method
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            logger.warning(
                f"{obj.__name__} is deprecated and will be removed in future versions. {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return obj(*args, **kwargs)
        return wrapper

    return decorator
