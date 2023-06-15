import ast
import os
import uuid
from abc import ABC
from json import JSONEncoder
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import core


class SymbolEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__


class Symbol(ABC):
    _dynamic_context: Dict[str, List[str]] = {}

    def __init__(self, *value) -> None:
        super().__init__()
        if len(value) == 1:
            value = value[0]
            if isinstance(value, Symbol):
                self.value = value.value
            elif isinstance(value, list) or isinstance(value, dict) or \
                    isinstance(value, set) or isinstance(value, tuple) or \
                        isinstance(value, str) or isinstance(value, int) or \
                            isinstance(value, float) or isinstance(value, bool):
                # unwrap nested symbols
                if isinstance(value, list):
                    value = [v.value if isinstance(v, Symbol) else v for v in value]
                elif isinstance(value, dict):
                    value = {k: v.value if isinstance(v, Symbol) else v for k, v in value.items()}
                elif isinstance(value, set):
                    value = {v.value if isinstance(v, Symbol) else v for v in value}
                elif isinstance(value, tuple):
                    value = tuple([v.value if isinstance(v, Symbol) else v for v in value])
                self.value = value
            else:
                self.value = value
        elif len(value) > 1:
            self.value = [v.value if isinstance(v, Symbol) else v for v in value]
        else:
            self.value = None

        self._static_context: str = ''

    @property
    def _sym_return_type(self):
        return Symbol

    @property
    def global_context(self) -> str:
        return (self.static_context, self.dynamic_context)

    @property
    def static_context(self) -> str:
        return self._static_context

    @property
    def dynamic_context(self) -> str:
        type_ = str(type(self))
        if type_ not in Symbol._dynamic_context:
            Symbol._dynamic_context[type_] = []
        _dyn_ctxt = Symbol._dynamic_context[type_]
        if len(_dyn_ctxt) == 0:
            return ''
        val_ = '\n'.join(_dyn_ctxt)
        val_ = f"\nSPECIAL RULES:\n{val_}"
        return val_

    def _to_symbol(self, value: Any) -> "Symbol":
        if isinstance(value, Symbol):
            return value
        return Symbol(value)

    def update(self, feedback: str) -> "Symbol":
        type_ = str(type(self))
        if type_ not in Symbol._dynamic_context:
            Symbol._dynamic_context[type_] = []
        self._dynamic_context[type_].append(feedback)
        return self

    def clear(self) -> "Symbol":
        type_ = str(type(self))
        if type_ not in Symbol._dynamic_context:
            Symbol._dynamic_context[type_] = []
            return self
        self._dynamic_context.clear()
        return self

    def __call__(self, *args, **kwargs):
        return self.value

    def __hash__(self) -> int:
        return str(self.value).__hash__()

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, key):
        if not self.__dict__.__contains__(key):
            try:
                att = getattr(self.value, key)
            except AttributeError as e:
                raise AttributeError(f"Cascading call failed, since object has no attribute '{key}'. Original error message: {e}")
            return att
        return self.__dict__[key]

    def __contains__(self, other) -> bool:
        @core.contains()
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def isinstanceof(self, query: str, **kwargs) -> bool:
        @core.isinstanceof()
        def _func(_, query: str, **kwargs) -> bool:
            pass
        return _func(self, query, **kwargs)

    def __eq__(self, other) -> bool:
        @core.equals()
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __matmul__(self, other) -> "Symbol":
        return self._sym_return_type(str(self) + str(other))

    def __rmatmul__(self, other) -> "Symbol":
        return self._sym_return_type(str(other) + str(self))

    def __imatmul__(self, other) -> "Symbol":
        self.value = Symbol(str(self) + str(other))
        return self

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __gt__(self, other) -> bool:
        @core.compare(operator = '>')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __lt__(self, other) -> bool:
        @core.compare(operator = '<')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __le__(self, other) -> bool:
        @core.compare(operator = '<=')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __ge__(self, other) -> bool:
        @core.compare(operator = '>=')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __len__(self):
        return len(str(self.value))

    def __bool__(self):
        return bool(self.value) if isinstance(self.value, bool) else False

    @property
    def length(self) -> int:
        return len(str(self.value))

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def tokens(self) -> int:
        return self.tokenizer().encode(str(self))

    @core.bind(engine='neurosymbolic', property='tokenizer')
    def tokenizer(self) -> object:
        pass

    def type(self):
        return type(self.value)

    def cast(self, type_):
        return type_(self.value)

    def ast(self):
        return ast.literal_eval(str(self.value))

    def __str__(self) -> str:
        if self.value is None:
            return str(None)
        elif isinstance(self.value, list) or isinstance(self.value, np.ndarray) or isinstance(self.value, tuple):
            return str([str(v) for v in self.value])
        elif isinstance(self.value, dict):
            return str({k: str(v) for k, v in self.value.items()})
        elif isinstance(self.value, set):
            return str({str(v) for v in self.value})
        else:
            return str(self.value)

    def __repr__(self):
        return f"{type(self)}(value={str(self.value)})"

    def _repr_html_(self):
        return f"""<div class="alert alert-success" role="alert">
  {str(self.value)}
</div>"""

    def __iter__(self):
        if isinstance(self.value, list) or isinstance(self.value, tuple) or isinstance(self.value, np.ndarray):
            return iter(self.value)
        return self.list('item').value.__iter__()

    def __reversed__(self):
        return reversed(list(self.__iter__()))

    def __next__(self) -> "Symbol":
        return next(self.__iter__())

    def __getitem__(self, key) -> "Symbol":
        try:
            if (isinstance(key, int) or isinstance(key, slice)) and (isinstance(self.value, list) or isinstance(self.value, tuple) or isinstance(self.value, np.ndarray)):
                return self.value[key]
            elif (isinstance(key, str) or isinstance(key, int)) and isinstance(self.value, dict):
                return self.value[key]
        except:
            pass
        @core.getitem()
        def _func(_, index: str):
            pass
        return self._sym_return_type(_func(self, key))

    def __setitem__(self, key, value):
        try:
            if (isinstance(key, int) or isinstance(key, slice)) and (isinstance(self.value, list) or isinstance(self.value, tuple) or isinstance(self.value, np.ndarray)):
                self.value[key] = value
                return
            elif (isinstance(key, str) or isinstance(key, int)) and isinstance(self.value, dict):
                self.value[key] = value
                return
        except:
            pass
        @core.setitem()
        def _func(_, index: str, value: str):
            pass
        self.value = Symbol(_func(self, key, value)).value

    def __delitem__(self, key):
        try:
            if (isinstance(key, str) or isinstance(key, int)) and isinstance(self.value, dict):
                del self.value[key]
                return
        except:
            pass
        @core.delitem()
        def _func(_, index: str):
            pass
        self.value = Symbol(_func(self, key)).value

    def __neg__(self) -> "Symbol":
        @core.negate()
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def __not__(self) -> "Symbol":
        @core.negate()
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def __invert__(self) -> "Symbol":
        @core.invert()
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def __lshift__(self, information) -> "Symbol":
        @core.include()
        def _func(_, information: str):
            pass
        return self._sym_return_type(_func(self, information))

    def __rshift__(self, information) -> "Symbol":
        @core.include()
        def _func(_, information: str):
            pass
        return self._sym_return_type(_func(self, information))

    def __rrshift__(self, information) -> "Symbol":
        @core.include()
        def _func(_, information: str):
            pass
        return self._sym_return_type(_func(self, information))

    def __add__(self, other) -> "Symbol":
        @core.combine()
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))

    def __radd__(self, other) -> "Symbol":
        @core.combine()
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(other, self))

    def __iadd__(self, other) -> "Symbol":
        self.value = self.__add__(other)
        return self

    def __sub__(self, other) -> "Symbol":
        @core.replace()
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._sym_return_type(_func(self, other, ''))

    def __rsub__(self, other) -> "Symbol":
        @core.replace()
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._sym_return_type(_func(other, self, ''))

    def __isub__(self, other) -> "Symbol":
        val = self.__sub__(other)
        self.value = val.value
        return self

    def __and__(self, other) -> "Symbol":
        @core.logic(operator='and')
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))

    def __or__(self, other) -> "Symbol":
        @core.logic(operator='or')
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))

    def __xor__(self, other) -> "Symbol":
        @core.logic(operator='xor')
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))

    def __truediv__(self, other) -> "Symbol":
        return self._sym_return_type(str(self).split(str(other)))

    def index(self, item: str, **kwargs) -> "Symbol":
        @core.getitem(**kwargs)
        def _func(_, item: str) -> int:
            pass
        return self._sym_return_type(_func(self, item))

    def equals(self, other: str, context: str = 'contextually', **kwargs) -> "Symbol":
        @core.equals(context=context, **kwargs)
        def _func(_, other: str) -> bool:
            pass
        return self._sym_return_type(_func(self, other))

    def expression(self, expr: Optional[str] = None, expression_engine: str = None, **kwargs) -> "Symbol":
        if expr is None:
            expr = self.value
        @core.expression(expression_engine=expression_engine, **kwargs)
        def _func(_, expr: str):
            pass
        return self._sym_return_type(_func(self, expr))

    def clean(self, **kwargs) -> "Symbol":
        @core.clean(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def summarize(self, context: Optional[str] = None, **kwargs) -> "Symbol":
        @core.summarize(context=context, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def outline(self, **kwargs) -> "Symbol":
        @core.outline(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def unique(self, keys: List[str] = [], **kwargs) -> "Symbol":
        @core.unique(keys=keys, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def compose(self, **kwargs) -> "Symbol":
        @core.compose(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def filter(self, criteria: str, include: bool = False, **kwargs) -> "Symbol":
        @core.filtering(criteria=criteria, include=include, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def modify(self, changes: str, **kwargs) -> "Symbol":
        @core.modify(changes=changes, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def replace(self, replace: str, value: str, **kwargs) -> "Symbol":
        @core.replace(**kwargs)
        def _func(_, replace: str, value: str):
            pass
        return self._sym_return_type(_func(self, replace, value))

    def remove(self, information: str, **kwargs) -> "Symbol":
        @core.replace(**kwargs)
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._sym_return_type(_func(self, information, ''))

    def include(self, information: str, **kwargs) -> "Symbol":
        @core.include(**kwargs)
        def _func(_, information: str):
            pass
        return self._sym_return_type(_func(self, information))

    def combine(self, sym: str, **kwargs) -> "Symbol":
        @core.combine(**kwargs)
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, sym))

    def rank(self, measure: str = 'alphanumeric', order: str = 'desc', **kwargs) -> "Symbol":
        @core.rank(order=order, **kwargs)
        def _func(_, measure: str) -> str:
            pass
        return self._sym_return_type(_func(self, measure))

    def extract(self, pattern: str, **kwargs) -> "Symbol":
        @core.extract(**kwargs)
        def _func(_, pattern: str) -> str:
            pass
        return self._sym_return_type(_func(self, pattern))

    def analyze(self, exception: Exception, query: Optional[str] = '', **kwargs) -> "Symbol":
        @core.analyze(exception=exception, query=query, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def correct(self, context: str, **kwargs) -> "Symbol":
        @core.correct(context=context, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def translate(self, language: str = 'English', **kwargs) -> "Symbol":
        @core.translate(language=language, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def choice(self, cases: List[str], default: str, **kwargs) -> "Symbol":
        @core.case(enum=cases, default=default, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def query(self, context: str, prompt: Optional[str] = None, examples = [], **kwargs) -> "Symbol":
        @core.query(context=context, prompt=prompt, examples=examples, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def convert(self, format: str, **kwargs) -> "Symbol":
        @core.convert(format=format, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def transcribe(self, modify: str, **kwargs) -> "Symbol":
        @core.transcribe(modify=modify, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def execute(self, **kwargs) -> "Symbol":
        @core.execute(**kwargs)
        def _func(_):
            pass
        return _func(self)

    def fexecute(self, **kwargs) -> "Symbol":
        def _func(sym: Symbol, **kargs):
            return sym.execute(**kargs)
        return self.ftry(_func, **kwargs)

    def simulate(self, **kwargs) -> "Symbol":
        @core.simulate(**kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def sufficient(self, query: str, **kwargs) -> "Symbol":
        @core.sufficient(query=query, **kwargs)
        def _func(_) -> bool:
            pass
        return self._sym_return_type(_func(self))

    def list(self, condition: str, **kwargs) -> "Symbol":
        @core.listing(condition=condition, **kwargs)
        def _func(_) -> list:
            pass
        return self._sym_return_type(_func(self))

    def contains(self, other, **kwargs) -> bool:
        @core.contains(**kwargs)
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def foreach(self, condition, apply, **kwargs) -> "Symbol":
        @core.foreach(condition=condition, apply=apply, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def map(self, **kwargs) -> "Symbol":
        assert isinstance(self.value, dict), "Map can only be applied to a dictionary"
        map_ = {}
        keys = []
        for v in self.value.values():
            k = Symbol(v).unique(keys, **kwargs)
            keys.append(k.value)
            map_[k.value] = v
        return self._sym_return_type(map_)

    def dict(self, context: str, **kwargs) -> "Symbol":
        @core.dictionary(context=context, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def template(self, template: str, placeholder = '{{placeholder}}', **kwargs) -> "Symbol":
        @core.template(template=template, placeholder=placeholder, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def style(self, description: str, libraries = [], template: str = None, placeholder: str = '{{placeholder}}', **kwargs) -> "Symbol":
        if template is None:
            template = self.value
        @core.style(description=description, libraries=libraries, template=template, placeholder=placeholder, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def cluster(self, **kwargs) -> "Symbol":
        @core.cluster(entries=self.value, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def embed(self, **kwargs) -> "Symbol":
        if not isinstance(self.value, list): self.value = [self.value]

        @core.embed(entries=self.value, **kwargs)
        def _func(_) -> list:
            pass
        return self._sym_return_type(_func(self))

    def similarity(self, other: 'Symbol', metric: str = 'cosine') -> float:
        def _ensure_format(x):
            if not isinstance(x, np.ndarray):
                if not isinstance(x, Symbol):
                    raise TypeError(f"Cannot compute similarity with type {type(x)}")
                x = np.array(x.value)
            return x.squeeze()[:, None]

        v = _ensure_format(self)
        o = _ensure_format(other)

        if metric == 'cosine':
            return (v.T@o / (v.T@v)**.5 * (o.T@o)**.5).item()
        else:
            raise NotImplementedError(f"Similarity metric {metric} not implemented. Available metrics: 'cosine'")

    def zip(self, **kwargs) -> List[Tuple[str, List, Dict]]:
        if not isinstance(self.value, str):
            raise ValueError(f'Expected id to be a string, got {type(self.value)}')

        embeds = self.embed(**kwargs).value
        idx    = str(uuid.uuid4())
        query  = {'text': self.value}

        return list(zip([idx], embeds, [query]))

    def stream(self, expr: "Expression",
               max_tokens: int = 4000,
               char_token_ratio: float = 0.6,
               **kwargs) -> "Symbol":
        max_chars = int(max_tokens * char_token_ratio)
        steps = (len(self)// max_chars) + 1
        for chunks in range(steps):
            # iterate over string in chunks of max_chars
            r = Symbol(str(self)[chunks * max_chars: (chunks + 1) * max_chars])
            size = max_tokens - r.size()

            # simulate the expression
            prev = expr(r, max_tokens=size, preview=True, **kwargs)
            # if the expression is too big, split it
            if prev.size() > max_tokens:
                # split
                r1_split = r.value[:len(r)//2]
                r = expr(r1_split, max_tokens=size, **kwargs)
                yield r
                r2_split = r.value[len(r)//2:]
                r = expr(r2_split, max_tokens=size, **kwargs)
            else:
                # run the expression
                r = expr(r, max_tokens=size, **kwargs)

            yield r

    def fstream(self, expr: "Expression",
                max_tokens: int = 4000,
                char_token_ratio: float = 0.6,
                **kwargs) -> "Symbol":
        return self._sym_return_type(list(self.stream(expr, max_tokens, char_token_ratio, **kwargs)))

    def ftry(self, expr: "Expression", retries: int = 1, **kwargs) -> "Symbol":
        prompt = {'message': ''}
        def output_handler(input_):
            prompt['message'] = input_
        kwargs['output_handler'] = output_handler
        retry_cnt: int = 0
        sym = self
        while True:
            try:
                sym = expr(sym, **kwargs)
                retry_cnt = 0
                return sym
            except Exception as e:
                retry_cnt += 1
                if retry_cnt > retries:
                    raise e
                else:
                    err =  Symbol(prompt['message'])
                    res = err.analyze(query="What is the issue in this expression?", payload=sym, exception=e, max_tokens=2000)
                    sym = sym.correct(context=prompt['message'], exception=e, payload=res, max_tokens=2000)

    def expand(self, *args, **kwargs) -> "Symbol":
        @core.expand(max_tokens=2048, **kwargs)
        def _func(_, *args):
            pass
        _tmp_llm_func = self._sym_return_type(_func(self, *args))
        func_name = str(_tmp_llm_func.extract('function name'))
        def _llm_func(*args, **kwargs):
            res = _tmp_llm_func.fexecute(*args, **kwargs)
            return res['locals'][func_name]()
        setattr(self, func_name, _llm_func)
        return func_name

    def draw(self, operation: str = 'create', **kwargs) -> "Symbol":
        @core.draw(operation=operation, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def save(self, path: str, replace: bool = False) -> "Symbol":
        file_path = path
        if not replace:
            cnt = 0
            while os.path.exists(file_path):
                filename, file_extension = os.path.splitext(path)
                file_path = f'{filename}_{cnt}{file_extension}'
                cnt += 1
        with open(file_path, 'w') as f:
            f.write(str(self))
        return self

    def output(self, *args, **kwargs) -> "Symbol":
        @core.output(**kwargs)
        def _func(_, *args):
            pass
        return self._sym_return_type(_func(self, *args))


class Expression(Symbol):
    def __init__(self, value = None):
        super().__init__(value)

    @property
    def _sym_return_type(self):
        return Expression

    def __call__(self, *args, **kwargs) -> Symbol:
        self.value = self.forward(*args, **kwargs)
        return self.value

    def forward(self, *args, **kwargs) -> Symbol:
        raise NotImplementedError()

    def input(self, message: str = "Please add more information", **kwargs) -> "Symbol":
        @core.userinput(**kwargs)
        def _func(_, message) -> str:
            pass
        return self._sym_return_type(_func(self, message))

    def fetch(self, url: str, pattern: str = '', **kwargs) -> "Symbol":
        @core.fetch(url=url, pattern=pattern, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def ocr(self, image_url: str, **kwargs) -> "Symbol":
        if not image_url.startswith('http'):
            image_url = f'file://{image_url}'
        @core.ocr(image=image_url, **kwargs)
        def _func(_) -> dict:
            pass
        return self._sym_return_type(_func(self))

    def vision(self, image: Optional[str] = None, text: Optional[List[str]] = None, **kwargs) -> "Symbol":
        @core.vision(image=image, prompt=text, **kwargs)
        def _func(_) -> np.ndarray:
            pass
        return self._sym_return_type(_func(self))

    def speech(self, audio_path: str, operation: str = 'decode', **kwargs) -> "Symbol":
        @core.speech(audio=audio_path, prompt=operation, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def search(self, query: str, **kwargs) -> "Symbol":
        @core.search(query=query, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def open(self, path: str, **kwargs) -> "Symbol":
        @core.opening(path=path, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def index(self, path: str, **kwargs) -> "Symbol":
        @core.index(prompt=path, operation='config', **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def add(self, query: str, **kwargs) -> "Symbol":
        @core.index(prompt=query, operation='add', **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def get(self, query: str, **kwargs) -> "Symbol":
        @core.index(prompt=query, operation='search', **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    @staticmethod
    def command(engines: List[str] = ['all'], **kwargs) -> "Symbol":
        @core.command(engines=engines, **kwargs)
        def _func(_):
            pass
        return Expression(_func(Expression()))

    @staticmethod
    def setup(engines: Dict[str, Any], **kwargs) -> "Symbol":
        @core.setup(engines=engines, **kwargs)
        def _func(_):
            pass
        return Expression(_func(Expression()))
