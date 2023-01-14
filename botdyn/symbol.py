import os
from abc import ABC
from json import JSONEncoder
from typing import Dict, Iterator, List, Any, Optional
import numpy as np
import botdyn as bd
from transformers import GPT2Tokenizer


tokenizer = None # lazy load tokenizer


class SymbolEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__


class Symbol(ABC):
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
                self.value = value
            else:
                self.value = value
        elif len(value) > 1:
            self.value = [v.value if isinstance(v, Symbol) else v for v in value]
        else:
            self.value = None
            
        self._static_context = []
            
    @property
    def _sym_return_type(self):
        return Symbol
    
    @property
    def static_context(self):
        if len(self._static_context) == 0:
            return ''
        val_ = '\n'.join(self._static_context)
        val_ = f"\nVERY Important Special Rules:\n{val_}"
        return val_

    def __call__(self, *args, **kwargs):
        return self.value
    
    def __hash__(self) -> int:
        return str(self.value).__hash__()
    
    def __getattr__(self, key):
        if not self.__dict__.__contains__(key):
            return getattr(self.value, key)
        return self.__dict__[key]
    
    @bd.contains()
    def __contains__(self, other) -> bool:
        pass
    
    @bd.isinstanceof()
    def isinstanceof(self, query: str, **kwargs) -> bool:
        pass
    
    @bd.equals()
    def __eq__(self, other) -> bool:
        pass
    
    def __matmul__(self, other) -> "Symbol":
        return self._sym_return_type(str(self) + str(other))
        
    def __rmatmul__(self, other) -> "Symbol":
        return self._sym_return_type(str(other) + str(self))
    
    def __imatmul__(self, other) -> "Symbol":
        self.value = Symbol(str(self) + str(other))
        return self
    
    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    @bd.compare(operator = '>')
    def __gt__(self, other) -> bool:
        pass
    
    @bd.compare(operator = '<')
    def __lt__(self, other) -> bool:
        pass
    
    @bd.compare(operator = '<=')
    def __le__(self, other) -> bool:
        pass
    
    @bd.compare(operator = '>=')
    def __ge__(self, other) -> bool:
        pass
    
    def __len__(self):
        return len(str(self.value))
    
    def __bool__(self):
        return bool(self.value) if isinstance(self.value, bool) else False
    
    @property
    def length(self) -> int:
        return len(str(self.value))
    
    def size(self) -> int:
        global tokenizer
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return len(tokenizer(str(self.value)).input_ids)
    
    def tokens(self) -> int:
        global tokenizer
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return tokenizer(str(self.value)).input_ids
    
    def type(self):
        return type(self.value)
    
    def cast(self, type_):
        return type_(self.value)
    
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
        return f"""<div class="alert alert-primary" role="alert">
  <h4 class="alert-heading">{str(self.value)}</h4>
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
        @bd.getitem()
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
        @bd.setitem()
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
        @bd.delitem()
        def _func(_, index: str):
            pass
        self.value = Symbol(_func(self, key)).value
    
    def __neg__(self) -> "Symbol":
        @bd.negate()
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def __not__(self) -> "Symbol":
        @bd.negate()
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def __invert__(self) -> "Symbol":
        @bd.invert()
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def __lshift__(self, other) -> "Symbol":
        @bd.include()
        def _func(_, text: str, information: str):
            pass
        return self._sym_return_type(_func(self, other))
    
    def __rshift__(self, other) -> "Symbol":
        @bd.include()
        def _func(_, text: str, information: str):
            pass
        return self._sym_return_type(_func(self, other))
    
    def __add__(self, other) -> "Symbol":
        @bd.combine()
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))
    
    def __radd__(self, other) -> "Symbol":
        @bd.combine()
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(other, self))
    
    def __iadd__(self, other) -> "Symbol":
        self.value = self.__add__(other)
        return self
    
    def __sub__(self, other) -> "Symbol":
        @bd.replace()
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._sym_return_type(_func(self, other, ''))
    
    def __rsub__(self, other) -> "Symbol":
        @bd.replace()
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._sym_return_type(_func(other, self, ''))
    
    def __isub__(self, other) -> "Symbol":
        val = self.__sub__(other)
        self.value = val.value
        return self
    
    def __and__(self, other) -> "Symbol":
        @bd.logic(operator='and')
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))
    
    def __or__(self, other) -> "Symbol":
        @bd.logic(operator='or')
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))
    
    def __xor__(self, other) -> "Symbol":
        @bd.logic(operator='xor')
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))
    
    def __truediv__(self, other) -> "Symbol":
        return self._sym_return_type(str(self).split(str(other)))
    
    def index(self, item: str, **kwargs) -> "Symbol":
        @bd.getitem(**kwargs)
        def _func(_, item: str) -> int:
            pass
        return self._sym_return_type(_func(self, item))
    
    def equals(self, other: str, context: str = 'contextually', **kwargs) -> "Symbol":
        @bd.equals(context=context, **kwargs)
        def _func(_, other: str) -> bool:
            pass
        return self._sym_return_type(_func(self, other))
    
    def expression(self, expr: Optional[str] = None, **kwargs) -> "Symbol":
        if expr is None:
            expr = 'self'
        @bd.expression(**kwargs)
        def _func(_, expr: str):
            pass
        return self._sym_return_type(_func(self, expr))
    
    def clean(self, **kwargs) -> "Symbol":
        @bd.clean(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def summarize(self, **kwargs) -> "Symbol":
        @bd.summarize(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def outline(self, **kwargs) -> "Symbol":
        @bd.outline(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def unique(self, keys: List[str] = [], **kwargs) -> "Symbol":
        @bd.unique(keys=keys, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def compose(self, **kwargs) -> "Symbol":
        @bd.compose(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def filter(self, criteria: str, include: bool = False, **kwargs) -> "Symbol":
        @bd.filtering(criteria=criteria, include=include, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def modify(self, changes: str, **kwargs) -> "Symbol":
        @bd.modify(changes=changes, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def replace(self, replace: str, value: str, **kwargs) -> "Symbol":
        @bd.replace(**kwargs)
        def _func(_, replace: str, value: str):
            pass
        return self._sym_return_type(_func(self, replace, value))
    
    def remove(self, information: str, **kwargs) -> "Symbol":
        @bd.replace(**kwargs)
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._sym_return_type(_func(self, information, ''))
    
    def include(self, information: str, **kwargs) -> "Symbol":
        @bd.include(**kwargs)
        def _func(_, text: str, information: str):
            pass
        return self._sym_return_type(_func(self, information))
    
    def combine(self, sym: str, **kwargs) -> "Symbol":
        @bd.combine(**kwargs)
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, sym))
    
    def rank(self, measure: str = 'alphanumeric', order: str = 'desc', **kwargs) -> "Symbol":
        @bd.rank(order=order, **kwargs)
        def _func(_, measure: str) -> str:
            pass
        return self._sym_return_type(_func(self, measure))
    
    def extract(self, pattern: str, **kwargs) -> "Symbol":
        @bd.extract(**kwargs)
        def _func(_, pattern: str, text: str) -> str:
            pass
        return self._sym_return_type(_func(self, pattern))
    
    def analyze(self, exception: Exception, query: Optional[str] = '', **kwargs) -> "Symbol":
        @bd.analyze(exception=exception, query=query, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def correct(self, context: str, **kwargs) -> "Symbol":
        @bd.correct(context=context, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def translate(self, language: str = 'English', **kwargs) -> "Symbol":
        @bd.translate(language=language, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def choice(self, cases: List[str], default: str, **kwargs) -> "Symbol":
        @bd.case(enum=cases, default=default, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def query(self, context: str, prompt: Optional[str] = None, examples = [], **kwargs) -> "Symbol":
        @bd.query(context=context, prompt=prompt, examples=examples, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def convert(self, format: str) -> "Symbol":
        @bd.convert(format=format)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def input(self, prompt: str = "Please add more information", **kwargs) -> "Symbol":
        @bd.userinput(prompt=prompt, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def execute(self, **kwargs) -> "Symbol":
        @bd.execute(**kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def simulate(self, **kwargs) -> "Symbol":
        @bd.simulate(**kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def sufficient(self, query: str, **kwargs) -> "Symbol":
        @bd.sufficient(query=query, **kwargs)
        def _func(_) -> bool:
            pass
        return self._sym_return_type(_func(self))
    
    def list(self, condition: str, **kwargs) -> "Symbol":
        @bd.listing(condition=condition, **kwargs)
        def _func(_) -> list:
            pass
        return self._sym_return_type(_func(self))
    
    def contains(self, other, **kwargs) -> bool:
        @bd.contains(**kwargs)
        def _func(_, other) -> bool:
            pass
        return _func(self, other)
    
    def foreach(self, condition, apply, **kwargs) -> "Symbol":
        @bd.foreach(condition=condition, apply=apply, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def map(self) -> "Symbol":
        assert isinstance(self.value, dict), "Map can only be applied to a dictionary"
        map_ = {}
        keys = []
        for v in self.value.values():
            k = Symbol(v).unique(keys)
            keys.append(k.value)
            map_[k.value] = v
        return self._sym_return_type(map_)
    
    def dict(self, context: str, **kwargs) -> "Symbol":
        @bd.dictionary(context=context, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def template(self, tempalte: str, placeholder='{{placeholder}}', **kwargs) -> "Symbol":
        @bd.template(template=tempalte, placeholder=placeholder, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def style(self, description: str, libraries = [], **kwargs) -> "Symbol":
        @bd.style(description=description, libraries=libraries, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def cluster(self, **kwargs) -> "Symbol":
        @bd.cluster(entries=self.value, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def embed(self, **kwargs) -> "Symbol":
        @bd.embed(entries=self.value, **kwargs)
        def _func(_) -> list:
            pass
        return self._sym_return_type(_func(self))
    
    def draw(self, operation: str = 'create', **kwargs) -> "Symbol":
        @bd.draw(operation=operation, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))
    
    def save(self, path: str, replace: bool = False, **kwargs) -> "Symbol":
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
        @bd.output(**kwargs)
        def _func(_, *args):
            pass
        return self._sym_return_type(_func(self, *args))
    
    def command(self, **kwargs) -> "Symbol":
        @bd.command(**kwargs)
        def _func(_):
            pass
        _func(self)
        return self
    
    def setup(self, engines: Dict[str, Any], **kwargs) -> "Symbol":
        @bd.setup(engines=engines, **kwargs)
        def _func(_):
            pass
        _func(self)
        return self
    
    def update(self, feedback: str, **kwargs) -> "Symbol":
        self._static_context.append(feedback)
        return self
    
    def clear(self, **kwargs) -> "Symbol":
        self._static_context.clear()
        return self


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
    
    def fetch(self, url: str, pattern: str = '', **kwargs) -> "Symbol":
        @bd.fetch(url=url, pattern=pattern, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def ocr(self, image_url: str, **kwargs) -> "Symbol":
        @bd.ocr(image=image_url, **kwargs)
        def _func(_) -> dict:
            pass
        return self._sym_return_type(_func(self))
    
    def vision(self, image: Optional[str] = None, text: Optional[List[str]] = None, **kwargs) -> "Symbol":
        @bd.vision(image=image, prompt=text, **kwargs)
        def _func(_) -> np.ndarray:
            pass
        return self._sym_return_type(_func(self))
    
    def speech(self, audio_path: str, operation: str = 'decode', **kwargs) -> "Symbol":
        @bd.speech(audio=audio_path, prompt=operation, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def search(self, query: str, **kwargs) -> "Symbol":
        @bd.search(query=query, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
    def open(self, path: str, **kwargs) -> "Symbol":
        @bd.opening(path=path, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
    
