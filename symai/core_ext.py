import functools
import traceback
import os
import os
import pickle
import random
import time
import logging
import dill
import multiprocessing as mp

from . import __root_dir__
from typing import Callable, List
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as PPool

from .functional import EngineRepository


logging.getLogger("multiprocessing").setLevel(logging.ERROR)


def _run_in_process(expr, func, args, kwargs):
    # Unpickling the expression using dill
    expr = dill.loads(expr)
    func = dill.loads(func)
    return func(expr, *args, **kwargs)

def _parallel(func: Callable, expressions: List[Callable], worker: int = mp.cpu_count() // 2):
    def proxy_function(*args, **kwargs):
        # Pickle the expressions using dill
        pickled_exprs = [dill.dumps(expr) for expr in expressions]
        pickled_func = dill.dumps(func)
        with mp.Pool(processes=worker) as pool:
            # We map the _run_in_process function to each pickled expression
            results = pool.starmap(_run_in_process, [(expr, pickled_func, args, kwargs) for expr in pickled_exprs])
        return results
    return proxy_function

# Decorator
def parallel(expressions: List[Callable], worker: int = mp.cpu_count() // 2):
    def decorator_parallel(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Run expressions in parallel
            parallel_func = _parallel(func, expressions, worker=worker)
            # Call the proxy function to execute in parallel and capture results
            results = parallel_func(*args, **kwargs)
            return results
        return wrapper
    return decorator_parallel

def bind(engine: str, property: str):
    '''
    Bind to an engine and retrieve one of its properties.
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
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
    jitter=0
):
    '''
    Returns a retry decorator.

    :param exceptions: an exception or a tuple of exceptions to catch. default: Exception.
    :param tries:      the maximum number of attempts. default: -1 (infinite).
    :param delay:      initial delay between attempts. default: 0.
    :param max_delay:  the maximum value of delay. default: -1 (no limit).
    :param backoff:    multiplier applied to delay between attempts. default: 1 (no backoff).
    :param jitter:     extra seconds added to delay between attempts. default: 0.
                       fixed if a number, random if a range tuple (min, max)

    Credits to invlpg (https://pypi.org/project/retry)
    '''

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _retry_func(
                    functools.partial(func, *args, **kwargs),
                    exceptions=exceptions,
                    tries=tries,
                    delay=delay,
                    max_delay=max_delay,
                    backoff=backoff,
                    jitter=jitter
                )
        return wrapper
    return decorator


def _retry_func(
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

    if not os.path.exists(cache_path): os.makedirs(cache_path)

    if in_memory and os.path.exists(Path(cache_path) / func.__qualname__):
        with open(Path(cache_path) / func.__qualname__, 'rb') as f:
            call = pickle.load(f)

        return call

    call = func(*args, **kwargs)
    with open(Path(cache_path) / func.__qualname__, 'wb') as f:
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
                logging.error(e)
                if debug:
                    # Simple message:
                    print('Function: {} call failed. Error: {}'.format(func.__name__, e))
                    # print traceback
                    traceback.print_exc()
                raise e
        return _dec
    return dec
