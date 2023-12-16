import inspect
import os
import re
import numpy as np

from pathlib import Path
from random import sample
from string import ascii_lowercase, ascii_uppercase
from typing import Callable, Iterator, List, Optional, Type
from tqdm import tqdm

from . import core
from . import core_ext
from .constraints import DictFormatConstraint
from .formatter import ParagraphFormatter
from .symbol import Expression, Symbol, Metadata
from .utils import CustomUserWarning
from .prompts import Prompt, JsonPromptTemplate
from .pre_processors import PreProcessor, JsonPreProcessor
from .post_processors import PostProcessor, JsonTruncatePostProcessor


class TrackerTraceable(Expression):
    pass


class Any(Expression):
    def __init__(self, *expr: List[Expression]):
        super().__init__()
        self.expr: List[Expression] = expr

    def forward(self, *args, **kwargs) -> Symbol:
        return self.sym_return_type(any([e() for e in self.expr(*args, **kwargs)]))


class All(Expression):
    def __init__(self, *expr: List[Expression]):
        super().__init__()
        self.expr: List[Expression] = expr

    def forward(self, *args, **kwargs) -> Symbol:
        return self.sym_return_type(all([e() for e in self.expr(*args, **kwargs)]))


class Try(Expression):
    def __init__(self, expr: Expression, retries: int = 1):
        super().__init__()
        self.expr: Expression = expr
        self.retries: int = retries

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.ftry(self.expr, retries=self.retries, **kwargs)


class Lambda(Expression):
    def __init__(self, callable: Callable):
        super().__init__()
        def _callable(*args, **kwargs):
            kw = {
                'args': args,
                'kwargs': kwargs,
            }
            return callable(kw)
        self.callable: Callable = _callable

    def forward(self, *args, **kwargs) -> Symbol:
        return self.callable(*args, **kwargs)


class Choice(Expression):
    def __init__(self, cases: List[str], default: Optional[str] = None):
        super().__init__()
        self.cases: List[str] = cases
        self.default: Optional[str] = default

    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.choice(cases=self.cases, default=self.default, *args, **kwargs)


class Output(Expression):
    def __init__(self, expr: Expression, handler: Callable, verbose: bool = False):
        super().__init__()
        self.expr: Expression = expr
        self.handler: Callable = handler
        self.verbose: bool = verbose

    def forward(self, *args, **kwargs) -> Expression:
        kwargs['verbose'] = self.verbose
        kwargs['handler'] = self.handler
        return self.output(expr=self.expr, *args, **kwargs)


class Sequence(TrackerTraceable):
    def __init__(self, *expr: List[Expression]):
        super().__init__()
        self.expr: List[Expression] = expr

    def forward(self, *args, **kwargs) -> Symbol:
        sym = self.expr[0](*args, **kwargs)
        metadata = Metadata()
        metadata.results = []
        metadata.results.append(sym)
        for e in self.expr[1:]:
            sym = e(sym, **kwargs)
            metadata.results.append(sym)
        sym._metadata = metadata
        return sym


#@TODO: BinPacker(format="...") -> ensure that data packages form a "bin" that's consistent (e.g. never break a sentence in the middle)
class Stream(Expression):
    def __init__(self, expr: Optional[Expression] = None, retrieval: Optional[str] = None):
        super().__init__()
        self.char_token_ratio:    float = 0.6
        self.expr: Optional[Expression] = expr
        self.retrieval:   Optional[str] = retrieval
        self._trace:               bool = False
        self._previous_frame            = None

    def forward(self, sym: Symbol, **kwargs) -> Iterator:
        sym = self._to_symbol(sym)

        if self._trace:
            local_vars = self._previous_frame.f_locals
            vals = []
            for key, var in local_vars.items():
                if isinstance(var, TrackerTraceable):
                    vals.append(var)

            if len(vals) == 1:
                self.expr = vals[0]
            else:
                raise ValueError(f"This component does either not inherit from TrackerTraceable or has an invalid number of component declarations: {len(vals)}! Only one component that inherits from TrackerTraceable is allowed in the with stream clause.")

        res = sym.stream(expr=self.expr,
                         char_token_ratio=self.char_token_ratio,
                         **kwargs)

        if self.retrieval is not None:
            res = list(res)
            if self.retrieval == 'all':
                return res
            if self.retrieval == 'longest':
                res = sorted(res, key=lambda x: len(x), reverse=True)
                return res[0]
            if self.retrieval == 'contains':
                res = [r for r in res if self.expr in r]
                return res
            raise ValueError(f"Invalid retrieval method: {self.retrieval}")

        return res

    def __enter__(self):
        self._trace = True
        self._previous_frame = inspect.currentframe().f_back
        return self

    def __exit__(self, type, value, traceback):
        self._trace = False


class Trace(Expression):
    def __init__(self, expr: Optional[Expression] = None, engines=['all']):
        super().__init__()
        self.expr: Expression = expr
        self.engines: List[str] = engines

    def forward(self, *args, **kwargs) -> Expression:
        Expression.command(verbose=True, engines=self.engines)
        res = self.expr(*args, **kwargs)
        Expression.command(verbose=False, engines=self.engines)
        return res

    def __enter__(self):
        Expression.command(verbose=True, engines=self.engines)
        if self.expr is not None:
            return self.expr.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        Expression.command(verbose=False, engines=self.engines)
        if self.expr is not None:
            return self.expr.__exit__(type, value, traceback)


class Analyze(Expression):
    def __init__(self, exception: Exception, query: Optional[str] = None):
        super().__init__()
        self.exception: Expression = exception
        self.query: Optional[str] = query

    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        return sym.analyze(exception=self.exception, query=self.query, *args, **kwargs)


class Log(Expression):
    def __init__(self, expr: Optional[Expression] = None, engines=['all']):
        super().__init__()
        self.expr: Expression = expr
        self.engines: List[str] = engines

    def forward(self, *args, **kwargs) -> Expression:
        Expression.command(logging=True, engines=self.engines)
        res = self.expr(*args, **kwargs)
        Expression.command(logging=False, engines=self.engines)
        return res

    def __enter__(self):
        Expression.command(logging=True, engines=self.engines)
        if self.expr is not None:
            return self.expr.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        Expression.command(logging=False, engines=self.engines)
        if self.expr is not None:
            return self.expr.__exit__(type, value, traceback)


class Template(Expression):
    def __init__(self, template: str = "<html><body>{{placeholder}}</body></html>", placeholder: str = '{{placeholder}}'):
        super().__init__()
        self.placeholder = placeholder
        self.template_ = template

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.template(template=self.template_, placeholder=self.placeholder, **kwargs)


class Metric(Expression):
    def __init__(self, normalize: bool = False, eps: float = 1e-8):
        super().__init__()
        self.normalize  = normalize
        self.eps        = eps

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        assert sym.value_type == np.ndarray or sym.value_type == list, 'Metric can only be applied to numpy arrays or lists.'
        if sym.value_type == list:
            sym._value = np.array(sym.value)
        # compute normalization between 0 and 1
        if self.normalize:
            if len(sym.value.shape) == 1:
                sym._value = sym.value[None, :]
            elif len(sym.value.shape) == 2:
                pass
            else:
                raise ValueError(f'Invalid shape: {sym.value.shape}')
            # normalize between 0 and 1 and sum to 1
            sym._value = np.exp(sym.value) / (np.exp(sym.value).sum() + self.eps)
        return sym


class Style(Expression):
    def __init__(self, description: str, libraries: List[str] = []):
        super().__init__()
        self.description: str = description
        self.libraries: List[str] = libraries

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.style(description=self.description, libraries=self.libraries, **kwargs)


class Query(TrackerTraceable):
    def __init__(self, prompt: str):
        super().__init__()
        self.prompt: str = prompt

    def forward(self, sym: Symbol, context: Symbol = None, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.query(prompt=self.prompt, context=context, **kwargs)


class Outline(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.outline(**kwargs)


class Clean(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.clean(**kwargs)


class Execute(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.execute(**kwargs)


class Convert(Expression):
    def __init__(self, format: str = 'Python'):
        super().__init__()
        self.format = format

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.convert(format=self.format, **kwargs)


class Embed(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.embed(**kwargs)


class Cluster(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.cluster(**kwargs)


class Compose(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.compose(**kwargs)


class Map(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.map(**kwargs)


class Translate(Expression):
    def __init__(self, language: str = 'English'):
        super().__init__()
        self.language = language

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        if sym.isinstanceof(f'{self.language} text'):
            return sym
        return sym.translate(language=self.language, **kwargs)


class IncludeFilter(Expression):
    def __init__(self, include: str):
        super().__init__()
        self.include = include

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.filter(self.include, include=True, **kwargs)


class ExcludeFilter(Expression):
    def __init__(self, exclude: str):
        super().__init__()
        self.exclude = exclude

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.filter(self.exclude, include=False, **kwargs)


class FileReader(Expression):
    @classmethod
    def exists(cls, path: str) -> bool:
        # remove slicing if any
        _tmp     = path
        _splits  = _tmp.split('[')
        if '[' in _tmp:
            _tmp = _splits[0]
        assert len(_splits) == 1 or len(_splits) == 2, 'Invalid file link format.'
        _tmp     = Path(_tmp)
        # check if file exists and is a file
        if os.path.exists(_tmp) and os.path.isfile(_tmp):
            return True
        return False

    @classmethod
    def extract_files(cls, cmds: str) -> List[str]:
        # Use the updated regular expression to match quoted and non-quoted paths
        pattern = r'''(?:"((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)'|`((?:\\.|[^`\\])*)`|((?:\\ |[^ ])+))'''

        # Use the updated regular expression to split and handle quoted and non-quoted paths
        matches = re.findall(pattern, cmds)

        # Process the matches to handle quoted paths and normal paths
        files = []
        for match in matches:
            # Each match will have 4 groups due to the pattern, only one will be non-empty
            quoted_double, quoted_single, quoted_backtick, non_quoted = match
            if quoted_double:
                # Remove backslashes used for escaping inside double quotes
                path = re.sub(r'\\(.)', r'\1', quoted_double)
                file = FileReader.expand_user_path(path)
                file = file.replace(' ', '\\ ')
                files.append(file)
            elif quoted_single:
                # Remove backslashes used for escaping inside single quotes
                path = re.sub(r'\\(.)', r'\1', quoted_single)
                file = FileReader.expand_user_path(path)
                file = file.replace(' ', '\\ ')
                files.append(file)
            elif quoted_backtick:
                # Remove backslashes used for escaping inside backticks
                path = re.sub(r'\\(.)', r'\1', quoted_backtick)
                files.append(FileReader.expand_user_path(path))
            elif non_quoted:
                # For non-quoted paths, we simply add them to the list after expanding
                files.append(FileReader.expand_user_path(non_quoted))

        return files

    @classmethod
    def expand_user_path(cls, path: str) -> str:
        return path.replace('~', os.path.expanduser('~'))

    def forward(self, path: str, **kwargs) -> Expression:
        return self.open(path, **kwargs)


class FileQuery(Expression):
    def __init__(self, path: str, filter: str):
        super().__init__()
        self.path = path
        file_open = FileReader()
        self.query_stream = Stream(Sequence(
            IncludeFilter(filter),
        ))
        self.file = file_open(path)

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        res = Symbol(list(self.query_stream(self.file)))
        return res.query(prompt=sym, context=res, **kwargs)


class Function(TrackerTraceable):
    def __init__(self, prompt: str,
                 examples: Optional[str] = [],
                 pre_processors: Optional[List[PreProcessor]] = None,
                 post_processors: Optional[List[PostProcessor]] = None,
                 default: Optional[object] = None,
                 constraints: List[Callable] = [],
                 return_type: Optional[Type] = str, *args, **kwargs):
        super().__init__()
        chars = ascii_lowercase + ascii_uppercase
        self.name = 'func_' + ''.join(sample(chars, 15))
        self.args = args
        self.kwargs = kwargs
        self._promptTemplate = prompt
        self._promptFormatArgs = []
        self._promptFormatKwargs = {}
        self.examples = Prompt(examples)
        self.pre_processors = pre_processors
        self.post_processors = post_processors
        self.constraints = constraints
        self.default = default
        self.return_type = return_type

    @property
    def prompt(self):
        # return a copy of the prompt template
        if len(self._promptFormatArgs) == 0 and len(self._promptFormatKwargs) == 0:
            return self._promptTemplate
        return f"{self._promptTemplate}".format(*self._promptFormatArgs,
                                                **self._promptFormatKwargs)

    def format(self, *args, **kwargs):
        self._promptFormatArgs = args
        self._promptFormatKwargs = kwargs

    def forward(self, *args, **kwargs) -> Expression:
        # special case for few shot function prompt definition override
        if 'fn' in kwargs:
            self.prompt = kwargs['fn']
            del kwargs['fn']
        @core.few_shot(prompt=self.prompt,
                  examples=self.examples,
                  pre_processors=self.pre_processors,
                  post_processors=self.post_processors,
                  constraints=self.constraints,
                  default=self.default,
                  *self.args, **self.kwargs)
        def _func(_, *args, **kwargs) -> self.return_type:
            pass
        _type = type(self.name, (Expression, ), {
            # constructor
            "forward": _func,
            "sym_return_type": Symbol,
            "static_context": self.static_context,
            "dynamic_context": self.dynamic_context,
        })
        obj = _type()
        obj.sym_return_type = _type

        return self._to_symbol(obj(*args, **kwargs))


class JsonParser(Expression):
    def __init__(self, query: str, json_: dict):
        super().__init__()
        func = Function(prompt=JsonPromptTemplate(query, json_),
                        constraints=[DictFormatConstraint(json_)],
                        pre_processors=[JsonPreProcessor()],
                        post_processors=[JsonTruncatePostProcessor()])
        self.fn = Try(func, retries=1)

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        res = self.fn(sym, **kwargs)
        return self._to_symbol(res.ast())


class SimilarityClassification(Expression):
    def __init__(self, classes: List[str], metric: str = 'cosine', in_memory: bool = False):
        super().__init__()
        self.classes   = classes
        self.metric    = metric
        self.in_memory = in_memory

        if self.in_memory:
            CustomUserWarning(f'Caching mode is enabled! It is your responsability to empty the .cache folder if you did changes to the classes. The cache is located at {Path.home()}/.symai/cache')

    def forward(self, x: Symbol) -> Symbol:
        x            = self._to_symbol(x)
        usr_embed    = x.embed()
        embeddings   = self._dynamic_cache()
        similarities = [usr_embed.similarity(emb, metric=self.metric) for emb in embeddings]
        similarities = sorted(zip(self.classes, similarities), key=lambda x: x[1], reverse=True)

        return Symbol(similarities[0][0])

    def _dynamic_cache(self):
        @core_ext.cache(in_memory=self.in_memory)
        def embed_classes(self):
            opts = map(Symbol, self.classes)
            embeddings = [opt.embed() for opt in opts]

            return embeddings

        return embed_classes(self)


class InContextClassification(Expression):
    def __init__(self, blueprint: Prompt):
        super().__init__()
        self.blueprint = blueprint

    def forward(self, x: Symbol, **kwargs) -> Symbol:
        @core.few_shot(
            prompt=x,
            examples=self.blueprint,
            **kwargs
        )
        def _func(_):
            pass

        return Symbol(_func(self))


class TokenTracker(Expression):
    def __init__(self):
        super().__init__()
        self._trace: bool    = False
        self._previous_frame = None

    @core_ext.bind(engine='neurosymbolic', property='max_tokens')
    def max_tokens(self): pass

    def __enter__(self):
        self._trace = True
        self._previous_frame = inspect.currentframe().f_back
        return self

    def __exit__(self, type, value, traceback):
        local_vars = self._previous_frame.f_locals
        vals = []
        for key, var in local_vars.items():
            if hasattr(var, 'token_ratio'):
                vals.append(var)

        for val in vals:
            max_ = self.max_tokens() * val.token_ratio
            print('\n\n================\n[Used tokens: {:.2f}%]\n================\n'.format(len(val) / max_ * 100))
        self._trace = False


class Indexer(Expression):
    DEFAULT = 'dataindex'

    @staticmethod
    def replace_special_chars(index: str):
        # replace special characters that are not for path
        index = str(index)
        index = index.replace('-', '')
        index = index.replace('_', '')
        index = index.replace(' ', '')
        index = index.lower()
        return index

    def __init__(self, index_name: str = DEFAULT, top_k: int = 8, batch_size: int = 20, formatter: Callable = ParagraphFormatter(), auto_add=False, raw_result=True):
        super().__init__()
        index_name = Indexer.replace_special_chars(index_name)
        self.index_name = index_name
        self.elements   = []
        self.batch_size = batch_size
        self.top_k      = top_k
        self.retrieval  = None
        self.formatter  = formatter
        self.raw_result = raw_result
        self.sym_return_type = Expression

        # append index name to indices.txt in home directory .symai folder (default)
        self.path = Path.home() / '.symai' / 'indices.txt'
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()
        if auto_add:
            self.register()

    def register(self):
        # check if index already exists in indices.txt and append if not
        change = False
        with open(self.path, 'r') as f:
            indices = f.read().split('\n')
            # filter out empty strings
            indices = [i for i in indices if i]
            if self.index_name not in indices:
                indices.append(self.index_name)
                change = True
        if change:
            with open(self.path, 'w') as f:
                f.write('\n'.join(indices))

    def exists(self) -> bool:
        # check if index exists in home directory .symai folder (default) indices.txt
        path = Path.home() / '.symai' / 'indices.txt'
        if not path.exists():
            return False
        with open(path, 'r') as f:
            indices = f.read().split('\n')
            if self.index_name in indices:
                return True

    def forward(self, data: Optional[Symbol] = None, raw_result: bool = False) -> Symbol:
        that = self
        if data is not None:
            data = self._to_symbol(data)
            # split text paragraph-wise and index each paragraph separately
            self.elements = self.formatter(data).value
            # run over the elments in batches
            for i in tqdm(range(0, len(self.elements), self.batch_size)):
                val = Symbol(self.elements[i:i+self.batch_size]).zip()
                that.add(val, index_name=that.index_name)

        def _func(query, *args, **kwargs):
            query_emb = Symbol(query).embed().value
            res = that.get(query_emb, index_name=that.index_name, index_top_k=that.top_k, ori_query=query, **kwargs)
            that.retrieval = res
            if that.raw_result or raw_result or ('raw_result' in kwargs and kwargs['raw_result']):
                return res
            rsp = res.query(query, *args, **kwargs)
            return rsp

        return _func
