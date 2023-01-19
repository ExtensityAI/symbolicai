import inspect
import ast
import traceback
from typing import Dict, List, Callable, Optional
from .prompts import Prompt
from .pre_processors import *
from .post_processors import *


# all engines are initialized after first usage
neurosymbolic_engine = None
_eval_symbolic_expressions_engine = None
symbolic_engine = None
ocr_engine = None
vision_engine = None
speech_engine = None
embedding_engine = None
userinput_engine = None
search_engine = None
crawler_engine = None
execute_engine = None
file_engine = None
output_engine = None
imagerendering_engine = None


class ConstraintViolationException(Exception):
    pass


def _execute_query(engine, post_processor, wrp_self, wrp_params, return_constraint, args, kwargs) -> List[object]:
    # build prompt and query engine
    engine.prepare(args, kwargs, wrp_params)
    rsp = engine(**wrp_params)[0] # currently only support single query
    if post_processor:
        for pp in post_processor:
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
        assert res is not None, "Return type cast failed! Check if the return type is correct or post_processor output matches desired format: " + str(rsp)
        rsp = res
    elif return_constraint == bool:
        # do not cast with bool -> always returns true
        rsp = str(rsp).lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh', 'ok']
    elif return_constraint == inspect._empty:
        pass        
    else:
        rsp = return_constraint(rsp)
    
    # check if satisfies constraints
    for constraint in wrp_params['constraints']:
        if not constraint(rsp):
            raise ConstraintViolationException("Constraint not satisfied:", res, constraint)
    
    return rsp


def _process_query(engine, 
                   wrp_self, 
                   func: Callable, 
                   prompt: str,
                   examples: Prompt, 
                   constraints: List[Callable] = [],
                   default: Optional[object] = None, 
                   limit: int = 1,
                   trials: int = 1,
                   pre_processor: Optional[List[PreProcessor]] = None,
                   post_processor: Optional[List[PostProcessor]] = None,
                   wrp_args = [], wrp_kwargs = {},
                   args = [], kwargs = {}):
    
    if pre_processor and not isinstance(pre_processor, list):
        pre_processor = [pre_processor]
    if post_processor and not isinstance(post_processor, list):
        post_processor = [post_processor]
    
    # check signature for return type
    sig = inspect.signature(func)
    return_constraint = sig._return_annotation
    assert 'typing' not in str(return_constraint), "Return type must be of base type not generic Typing object, e.g. int, str, list, etc."
    
    # prepare wrapper parameters
    wrp_params = {
        'wrp_self': wrp_self,
        'func': func,
        'prompt': prompt,
        'examples': examples,
        'constraints': constraints,
        'default': default,
        'limit': limit,
        'signature': sig,
        **wrp_kwargs
    }
    # remove nested wrp_kwargs #TODO verify why this is needed
    if 'wrp_kwargs' in wrp_params:
        for k, v in wrp_params['wrp_kwargs'].items():
            wrp_params[k] = v
    
    # pre-process text
    suffix = ''
    if pre_processor:
        for pp in pre_processor:
            t = pp(wrp_self, wrp_params, *args, **kwargs)
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
    try_cnt = 0
    while try_cnt < trials:
        try_cnt += 1
        try:
            rsp = _execute_query(engine, post_processor, wrp_self, wrp_params, return_constraint, args, kwargs)
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
            rsp = func(wrp_self, *args, **kwargs)
            # if there is also no default implementation, raise exception
            if rsp is None and wrp_params['default'] is None:
                raise e # raise exception if no default and no function implementation
            elif rsp is None: # return default if there is one
                rsp = wrp_params['default']

    # return based on return type
    limit_ = wrp_params['limit'] if wrp_params['limit'] is not None else len(rsp)
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


def check_or_init_neurosymbolic_func(engine = None):
    global neurosymbolic_engine
    if engine is not None:
        neurosymbolic_engine = engine
    elif neurosymbolic_engine is None:
        from .backend.engine_gpt3 import GPT3Engine
        neurosymbolic_engine = GPT3Engine()


def few_shot_func(wrp_self, 
                  func: Callable, 
                  prompt: str,
                  examples: Prompt, 
                  constraints: List[Callable] = [],
                  default: Optional[object] = None, 
                  limit: int = 1,
                  trials: int = 1,
                  pre_processor: Optional[List[PreProcessor]] = None,
                  post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
                  wrp_args = [], wrp_kwargs = [],
                  args = [], kwargs = []):
    check_or_init_neurosymbolic_func()
    return _process_query(engine=neurosymbolic_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=prompt,
                          examples=examples,
                          constraints=constraints,
                          default=default,
                          limit=limit,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)
    
    
def check_or_init_symbolic_func(engine = None):
    global symbolic_engine
    if engine is not None:
        symbolic_engine = engine
    elif symbolic_engine is None:
        from .backend.engine_wolframalpha import WolframAlphaEngine
        symbolic_engine = WolframAlphaEngine()


def symbolic_func(wrp_self, 
                  func: Callable, 
                  prompt: str,
                  examples: Prompt, 
                  constraints: List[Callable] = [],
                  default: Optional[object] = None, 
                  limit: int = 1,
                  trials: int = 1,
                  pre_processor: Optional[List[PreProcessor]] = None,
                  post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
                  wrp_args = [], wrp_kwargs = [],
                  args = [], kwargs = []):
    check_or_init_symbolic_func()
    return _process_query(engine=symbolic_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=prompt,
                          examples=examples,
                          constraints=constraints,
                          default=default,
                          limit=limit,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_search_func(engine = None):
    global search_engine
    if engine is not None:
        search_engine = engine
    elif search_engine is None:
        from .backend.engine_google import GoogleEngine
        search_engine = GoogleEngine()
        

def search_func(wrp_self, 
                func: Callable,
                query: str,
                constraints: List[Callable] = [],
                default: Optional[object] = None, 
                limit: int = 1,
                trials: int = 1,
                pre_processor: Optional[List[PreProcessor]] = None,
                post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
                wrp_args = [], wrp_kwargs = [],
                args = [], kwargs = []):
    check_or_init_search_func()
    wrp_kwargs['query'] = query
    return _process_query(engine=search_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=query,
                          examples=None,
                          constraints=constraints,
                          default=default,
                          limit=limit,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_open_func(engine = None):
    global file_engine
    if engine is not None:
        file_engine = engine
    elif file_engine is None:
        from .backend.engine_file import FileEngine
        file_engine = FileEngine()

    
def open_func(wrp_self, 
              func: Callable,
              path: str,
              constraints: List[Callable] = [],
              default: Optional[object] = None, 
              limit: Optional[int] = None,
              trials: int = 1,
              pre_processor: Optional[List[PreProcessor]] = None,
              post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
              wrp_args = [], wrp_kwargs = [],
              args = [], kwargs = []):
    check_or_init_open_func()
    wrp_kwargs['path'] = path
    return _process_query(engine=file_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=path,
                          examples=None,
                          constraints=constraints,
                          default=default,
                          limit=limit,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_output_func(engine = None):
    global output_engine
    if engine is not None:
        output_engine = engine
    elif output_engine is None:
        from .backend.engine_output import OutputEngine
        output_engine = OutputEngine()

    
def output_func(wrp_self, 
                func: Callable,
                constraints: List[Callable] = [],
                default: Optional[object] = None, 
                limit: Optional[int] = None,
                trials: int = 1,
                pre_processor: List[PreProcessor] = [ConsolePreProcessor()],
                post_processor: Optional[List[PostProcessor]] = [ConsolePostProcessor()],
                wrp_args = [], wrp_kwargs = [],
                args = [], kwargs = []):
    check_or_init_output_func()
    return _process_query(engine=output_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=None,
                          examples=None,
                          constraints=constraints,
                          default=default,
                          limit=limit,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_crawler_func(engine = None):
    global crawler_engine
    if engine is not None:
        crawler_engine = engine
    elif crawler_engine is None:
        from .backend.engine_crawler import CrawlerEngine
        crawler_engine = CrawlerEngine()


def crawler_func(wrp_self, 
                 func: Callable,
                 url: str,
                 pattern: str,
                 constraints: List[Callable] = [],
                 default: Optional[object] = None, 
                 limit: int = 1,
                 trials: int = 1,
                 pre_processor: Optional[List[PreProcessor]] = None,
                 post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
                 wrp_args = [], wrp_kwargs = [],
                 args = [], kwargs = []):
    check_or_init_crawler_func()
    wrp_kwargs['url'] = url
    wrp_kwargs['pattern'] = pattern
    return _process_query(engine=crawler_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=None,
                          examples=None,
                          constraints=constraints,
                          default=default,
                          limit=limit,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_userinput_func(engine = None):
    global userinput_engine
    if engine is not None:
        userinput_engine = engine
    elif userinput_engine is None:
        from .backend.engine_userinput import UserInputEngine
        userinput_engine = UserInputEngine()


def userinput_func(wrp_self, 
                   func: Callable,
                   prompt: Optional[str] = None,
                   constraints: List[Callable] = [],
                   default: Optional[object] = None, 
                   trials: int = 1,
                   pre_processor: Optional[List[PreProcessor]] = None,
                   post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
                   wrp_args = [], wrp_kwargs = [],
                   args = [], kwargs = []):
    check_or_init_userinput_func()
    return _process_query(engine=userinput_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=prompt,
                          examples=None,
                          constraints=constraints,
                          default=default,
                          limit=1,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_execute_func(engine = None):
    global execute_engine
    if engine is not None:
        execute_engine = engine
    elif execute_engine is None:
        from .backend.engine_python import PythonEngine
        execute_engine = PythonEngine()


def execute_func(wrp_self,
                 code: str,
                 func: Callable,
                 constraints: List[Callable] = [],
                 default: Optional[object] = None, 
                 trials: int = 1,
                 pre_processor: List[PreProcessor] = [],
                 post_processor: Optional[List[PostProcessor]] = [],
                 wrp_args = [], wrp_kwargs = [],
                 args = [], kwargs = []):
    check_or_init_execute_func()
    return _process_query(engine=execute_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=code,
                          examples=None,
                          constraints=constraints,
                          default=default,
                          limit=1,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_embedding_func(engine = None):
    global embedding_engine
    if engine is not None:
        embedding_engine = engine
    elif embedding_engine is None:
        from .backend.engine_embedding import EmbeddingEngine
        embedding_engine = EmbeddingEngine()


def embed_func(wrp_self,
               entries: List[str],
               func: Callable,
               trials: int = 1,
               pre_processor: Optional[List[PreProcessor]] = None,
               post_processor: Optional[List[PostProcessor]] = None,
               wrp_args = [], wrp_kwargs = [],
               args = [], kwargs = []):
    check_or_init_embedding_func()
    wrp_kwargs['entries'] = entries
    return _process_query(engine=embedding_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=None,
                          examples=None,
                          constraints=[],
                          default=None,
                          limit=1,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_imagerendering_func(engine = None):
    global imagerendering_engine
    if engine is not None:
        imagerendering_engine = engine
    elif imagerendering_engine is None:
        from .backend.engine_imagerendering import ImageRenderingEngine
        imagerendering_engine = ImageRenderingEngine()

    
def imagerendering_func(wrp_self,
                        func: Callable,
                        operation: str = 'create',
                        prompt: str = '',
                        trials: int = 1,
                        pre_processor: Optional[List[PreProcessor]] = None,
                        post_processor: Optional[List[PostProcessor]] = None,
                        wrp_args = [], wrp_kwargs = [],
                        args = [], kwargs = []):
    check_or_init_imagerendering_func()
    wrp_kwargs['operation'] = operation
    return _process_query(engine=imagerendering_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=prompt,
                          examples=None,
                          constraints=[],
                          default=None,
                          limit=1,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_ocr_func(engine = None):
    global ocr_engine
    if engine is not None:
        ocr_engine = engine
    elif ocr_engine is None:
        from .backend.engine_ocr import OCREngine
        ocr_engine = OCREngine()


def ocr_func(wrp_self, 
             image: str,
             func: Callable,
             trials: int = 1,
             pre_processor: Optional[List[PreProcessor]] = None,
             post_processor: Optional[List[PostProcessor]] = None,
             wrp_args = [], wrp_kwargs = [],
             args = [], kwargs = []):
    check_or_init_ocr_func()
    wrp_kwargs['image'] = image
    return _process_query(engine=ocr_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=None,
                          examples=None,
                          constraints=[],
                          default=None,
                          limit=1,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_vision_func(engine = None):
    global vision_engine
    if engine is not None:
        vision_engine = engine
    elif vision_engine is None:
        from .backend.engine_clip import CLIPEngine
        vision_engine = CLIPEngine()


def vision_func(wrp_self, 
                func: Callable,
                image: Optional[str] = None,
                prompt: Optional[str] = None,
                trials: int = 1,
                pre_processor: Optional[List[PreProcessor]] = None,
                post_processor: Optional[List[PostProcessor]] = None,
                wrp_args = [], wrp_kwargs = [],
                args = [], kwargs = []):
    check_or_init_vision_func()
    wrp_kwargs['image'] = image
    return _process_query(engine=vision_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=prompt,
                          examples=None,
                          constraints=[],
                          default=None,
                          limit=1,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def check_or_init_speech_func(engine = None):
    global speech_engine
    if engine is not None:
        speech_engine = engine
    elif speech_engine is None:
        from .backend.engine_speech import WhisperEngine
        speech_engine = WhisperEngine()

    
def speech_func(wrp_self, 
                func: Callable,
                trials: int = 1,
                prompt='decode',
                pre_processor: Optional[List[PreProcessor]] = None,
                post_processor: Optional[List[PostProcessor]] = None,
                wrp_args = [], wrp_kwargs = [],
                args = [], kwargs = []):
    check_or_init_speech_func()
    return _process_query(engine=speech_engine,
                          wrp_self=wrp_self,
                          func=func,
                          prompt=prompt,
                          examples=None,
                          constraints=[],
                          default=None,
                          limit=1,
                          trials=trials,
                          pre_processor=pre_processor,
                          post_processor=post_processor,
                          wrp_args=wrp_args,
                          wrp_kwargs=wrp_kwargs,
                          args=args,
                          kwargs=kwargs)


def command_func(wrp_self,
                 func: Callable,
                 engines: List[str] = ['all'],
                 wrp_kwargs = [],
                 kwargs = []):
    # prepare wrapper parameters
    wrp_params = {
        'wrp_self': wrp_self,
        'func': func,
        **kwargs,
        **wrp_kwargs
    }
    
    if 'all' in engines or 'neurosymbolic' in engines:
        check_or_init_neurosymbolic_func()
        neurosymbolic_engine.command(wrp_params)
    if 'all' in engines or 'symbolic' in engines:
        check_or_init_symbolic_func()
        symbolic_engine.command(wrp_params)
    if 'all' in engines or 'ocr' in engines:
        check_or_init_ocr_func()
        ocr_engine.command(wrp_params)
    if 'all' in engines or 'vision' in engines:
        check_or_init_vision_func()
        vision_engine.command(wrp_params)
    if 'all' in engines or 'speech' in engines:
        check_or_init_speech_func()
        speech_engine.command(wrp_params)
    if 'all' in engines or 'embedding' in engines:
        check_or_init_embedding_func()
        embedding_engine.command(wrp_params)
    if 'all' in engines or 'userinput' in engines:
        check_or_init_userinput_func()
        userinput_engine.command(wrp_params)
    if 'all' in engines or 'search' in engines:
        check_or_init_search_func()
        search_engine.command(wrp_params)
    if 'all' in engines or 'crawler' in engines:
        check_or_init_crawler_func()
        crawler_engine.command(wrp_params)
    if 'all' in engines or 'execute' in engines:
        check_or_init_execute_func()
        execute_engine.command(wrp_params)
    if 'all' in engines or 'open' in engines:
        check_or_init_open_func()
        file_engine.command(wrp_params)
    if 'all' in engines or 'output' in engines:
        check_or_init_output_func()
        output_engine.command(wrp_params)
    if 'all' in engines or 'imagerendering' in engines:
        check_or_init_imagerendering_func()
        imagerendering_engine.command(wrp_params)


def setup_func(wrp_self,
               func: Callable,
               engines: Dict[str, Any],
               wrp_kwargs = [],
               kwargs = []):
    if 'neurosymbolic' in engines:
        check_or_init_neurosymbolic_func(engine=engines['neurosymbolic'])
    if 'symbolic' in engines:
        check_or_init_symbolic_func(engine=engines['symbolic'])
    if 'ocr' in engines:
        check_or_init_ocr_func(engine=engines['ocr'])
    if 'vision' in engines:
        check_or_init_vision_func(engine=engines['vision'])
    if 'speech' in engines:
        check_or_init_speech_func(engine=engines['speech'])
    if 'embedding' in engines:
        check_or_init_embedding_func(engine=engines['embedding'])
    if 'userinput' in engines:
        check_or_init_userinput_func(engine=engines['userinput'])
    if 'search' in engines:
        check_or_init_search_func(engine=engines['search'])
    if 'crawler' in engines:
        check_or_init_crawler_func(engine=engines['crawler'])
    if 'execute' in engines:
        check_or_init_execute_func(engine=engines['execute'])
    if 'open' in engines:
        check_or_init_open_func(engine=engines['open'])
    if 'output' in engines:
        check_or_init_output_func(engine=engines['output'])
    if 'imagerendering' in engines:
        check_or_init_imagerendering_func(engine=engines['imagerendering'])
