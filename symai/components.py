import inspect
import sys
from pathlib import Path
from random import sample
from string import ascii_lowercase, ascii_uppercase
from typing import Callable, Iterator, List, Optional, Type

from tqdm import tqdm

from .backend.engine_embedding import EmbeddingEngine
from .backend.engine_gptX_chat import GPTXChatEngine
from .backend.engine_pinecone import IndexEngine
from .backend.mixin.openai import SUPPORTED_MODELS
from .constraints import DictFormatConstraint
from .core import *
from .formatter import ParagraphFormatter
from .symbol import Expression, Symbol
from .utils import CustomUserWarning


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
        for e in self.expr[1:]:
            sym = e(sym, **kwargs)
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
        return sym.template(self.template_, self.placeholder, **kwargs)


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
        @few_shot(prompt=self.prompt,
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
        @cache(in_memory=self.in_memory)
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
        @few_shot(
            prompt=x,
            examples=self.blueprint,
            **kwargs
        )
        def _func(_):
            pass

        return Symbol(_func(self))


class OpenAICostTracker:
    _supported_models = SUPPORTED_MODELS

    def __init__(self):
        self._inputs     = []
        self._outputs    = []
        self._embeddings = []
        self._zero_shots = 0
        self._few_shots  = 0

    def __enter__(self):
        if self._neurosymbolic_model() not in self._supported_models:
            CustomUserWarning(f'We are currently supporting only the following models for the {self.__class__.__name__} feature: {self._supported_models}. Any other model will simply be ignored.')
        sys.settrace(self._trace_call)

        return self

    def __exit__(self, *args):
        sys.settrace(None)

    def __repr__(self):
        return f'''
[BREAKDOWN]
{'-=-' * 13}

{self._neurosymbolic_model()} usage:
    ${self._compute_io_costs():.3f} for {sum(self._inputs)} input tokens and {sum(self._outputs)} output tokens

{self._embedding_model()} usage:
    ${self._compute_embedding_costs():.3f} for {sum(self._embeddings)} tokens

Total:
    ${self._compute_io_costs() + self._compute_embedding_costs():.3f}

{'-=-' * 13}

Zero-shot calls: {self._zero_shots}

{'-=-' * 13}

Few-shot calls: {self._few_shots}

{'-=-' * 13}
'''

    def _trace_call(self, frame, event, arg):
        if event != 'call': return

        code      = frame.f_code
        func_name = code.co_name

        if func_name != '_execute_query':
            if    func_name == 'zero_shot': self._zero_shots += 1
            elif  func_name == 'few_shot': self._few_shots += 1
            else: return

        engine = frame.f_locals.get('engine')

        if isinstance(engine, GPTXChatEngine):
            if self._neurosymbolic_model() not in self._supported_models: return

            inp      = ''
            prompt   = frame.f_locals['wrp_params'].get('prompt')
            examples = frame.f_locals['wrp_params'].get('examples')

            if prompt is not None:
                if isinstance(prompt, str): inp += prompt + '\n'

            if examples is not None:
                if    isinstance(examples, str): inp += examples
                elif  isinstance(examples, list): inp += '\n'.join(examples)
                elif  isinstance(examples, Prompt): inp += examples.__repr__()

            self._inputs.append(len(Symbol(inp).tokens))

        elif isinstance(engine, EmbeddingEngine):
            if self._embedding_model() not in self._supported_models: return

            text = frame.f_locals.get('wrp_self')

            if text is not None:
                if   isinstance(text, str): self._embeddings.append(len(Symbol(text).tokens))
                elif isinstance(text, list): self._embeddings.append(len(Symbol(text[0]).tokens))
                elif isinstance(text, Symbol): self._embeddings.append(len(text.tokens))

        return self._trace_return

    def _trace_return(self, frame, event, arg):
        if event != 'return': return

        engine = frame.f_locals.get('engine')

        if isinstance(engine, GPTXChatEngine):
            self._outputs.append(len(Symbol(arg).tokens))

    def _compute_io_costs(self):
        if self._neurosymbolic_model() not in self._supported_models: return 0

        return (sum(self._inputs) * self._neurosymbolic_pricing()['input']) + (sum(self._outputs) * self._neurosymbolic_pricing()['output'])

    def _compute_embedding_costs(self):
        if self._embedding_model() not in self._supported_models: return 0

        return sum(self._embeddings) * self._embedding_pricing()['usage']

    @bind(engine='neurosymbolic', property='model')
    def _neurosymbolic_model(self): pass

    @bind(engine='neurosymbolic', property='pricing')
    def _neurosymbolic_pricing(self): pass

    @bind(engine='embedding', property='model')
    def _embedding_model(self): pass

    @bind(engine='embedding', property='pricing')
    def _embedding_pricing(self): pass


class TokenTracker(Expression):
    def __init__(self):
        super().__init__()
        self._trace: bool    = False
        self._previous_frame = None

    @bind(engine='neurosymbolic', property='max_tokens')
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

    class IndexResult(Expression):
        def __init__(self, res, query: str):
            self.query = query
            self.raw   = res
            self.value = [v['metadata']['text'] for v in res['matches']]

        def _unpack_matches(self):
            for i, match in enumerate(self.value):
                match = match.strip()
                if match.startswith('# ----[FILE_START]') and '# ----[FILE_END]' in match:
                    m = match.split('[FILE_CONTENT]:')[-1].strip()
                    content, file_name = m.split('# ----[FILE_END]')
                    yield file_name.strip(), content.strip()
                else:
                    yield i+1, match

        def __str__(self):
            str_view = ''
            for filename, content in self._unpack_matches():
                # indent each line of the content
                content = '\n'.join(['  ' + line for line in content.split('\n')])
                str_view += f'* {filename}\n{content}\n\n'
            return f'''
[RESULT]
{'-=-' * 13}

Query: {self.query}

{'-=-' * 13}

Matches:

{str_view}
{'-=-' * 13}
'''

        def _repr_html_(self) -> str:
            # return a nicely styled HTML list results based on retrieved documents
            doc_str = ''
            for filename, content in self._unpack_matches():
                doc_str += f'<li><a href="{filename}"><b>{filename}</a></b><br>{content}</li>\n'
            return f'<ul>{doc_str}</ul>'

    def replace_special_chars(self, index: str):
        # replace special characters that are not for path
        index = str(index)
        index = index.replace('-', '')
        index = index.replace('_', '')
        index = index.replace(' ', '')
        index = index.lower()
        return index

    def __init__(self, index_name: str = DEFAULT, top_k: int = 8, batch_size: int = 20, formatter: Callable = ParagraphFormatter(), auto_add=True):
        super().__init__()
        index_name = self.replace_special_chars(index_name)
        self.index_name = index_name
        self.elements   = []
        self.batch_size = batch_size
        self.top_k      = top_k
        self.retrieval  = None
        self.formatter  = formatter
        self.sym_return_type = Expression

        Expression.setup({'index': IndexEngine(index_name=index_name)})
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
                that.add(val)

        def _func(query, *args, **kwargs):
            try:
                query_emb = Symbol(query).embed().value
                res = that.get(query_emb, index_top_k=that.top_k)
                res = res.ast()
            except Exception as e:
                message = ['Sorry, failed to interact with index. Please check index name and try again later:', str(e)]
                # package the message for the IndexResult class
                res = {'matches': [{'metadata': {'text': '\n'.join(message)}}]}
                return that.IndexResult(res, query)
            if ('raw_result' in kwargs and kwargs['raw_result']) or raw_result:
                that.retrieval = res
                return that.IndexResult(res, query)
            res = [v['metadata']['text'] for v in res['matches']]
            that.retrieval = res
            sym = that._to_symbol(res)
            rsp = sym.query(query, *args, **kwargs)
            return rsp

        return _func
