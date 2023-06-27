import sys
from pathlib import Path
from random import sample
from string import ascii_lowercase, ascii_uppercase
from typing import Callable, Iterator, List, Optional
from tqdm import tqdm
import re

from .backend.engine_embedding import EmbeddingEngine
from .backend.engine_gptX_chat import GPTXChatEngine
from .backend.mixin.openai import SUPPORTED_MODELS
from .core import *
from .symbol import Expression, Symbol
from .utils import CustomUserWarning
from .memory import VectorDatabaseMemory


class Any(Expression):
    def __init__(self, *expr: List[Expression]):
        super().__init__()
        self.expr: List[Expression] = expr

    def forward(self, *args, **kwargs) -> Symbol:
        return self._sym_return_type(any([e() for e in self.expr(*args, **kwargs)]))


class All(Expression):
    def __init__(self, *expr: List[Expression]):
        super().__init__()
        self.expr: List[Expression] = expr

    def forward(self, *args, **kwargs) -> Symbol:
        return self._sym_return_type(all([e() for e in self.expr(*args, **kwargs)]))


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


class Sequence(Expression):
    def __init__(self, *expr: List[Expression]):
        super().__init__()
        self.expr: List[Expression] = expr

    def forward(self, *args, **kwargs) -> Symbol:
        sym = self.expr[0](*args, **kwargs)
        for e in self.expr[1:]:
            sym = e(sym, **kwargs)
        return sym


class Stream(Expression):
    def __init__(self, expr: Expression, force: bool = False, max_tokens: int = 4000): #4097
        super().__init__()
        self.max_tokens: int = max_tokens
        self.char_token_ratio: float = 0.6
        self.expr: Expression = expr
        self.force: bool = force

    def forward(self, sym: Symbol, **kwargs) -> Iterator[Symbol]:
        sym = self._to_symbol(sym)
        if self.force:
            return sym.fstream(expr=self.expr,
                               max_tokens=self.max_tokens,
                               char_token_ratio=self.char_token_ratio,
                               **kwargs)
        return sym.stream(expr=self.expr,
                          max_tokens=self.max_tokens,
                          char_token_ratio=self.char_token_ratio,
                          **kwargs)


class Trace(Expression):
    def __init__(self, expr: Expression, engines=['all']):
        super().__init__()
        self.expr: Expression = expr
        self.engines: List[str] = engines

    def forward(self, *args, **kwargs) -> Expression:
        Expression.command(verbose=True, engines=self.engines)
        res = self.expr(*args, **kwargs)
        Expression.command(verbose=False, engines=self.engines)
        return res


class Analyze(Expression):
    def __init__(self, exception: Exception, query: Optional[str] = None):
        super().__init__()
        self.exception: Expression = exception
        self.query: Optional[str] = query

    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        return sym.analyze(exception=self.exception, query=self.query, *args, **kwargs)


class Log(Expression):
    def __init__(self, expr: Expression, engines=['all']):
        super().__init__()
        self.expr: Expression = expr
        self.engines: List[str] = engines

    def forward(self, *args, **kwargs) -> Expression:
        Expression.command(logging=True, engines=self.engines)
        res = self.expr(*args, **kwargs)
        Expression.command(logging=False, engines=self.engines)
        return res


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


class Query(Expression):
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


class Function(Expression):
    def __init__(self, prompt: str, static_context: str = "",
                 examples: Optional[str] = [],
                 pre_processor: Optional[List[PreProcessor]] = None,
                 post_processor: Optional[List[PostProcessor]] = None,
                 default: Optional[object] = None, *args, **kwargs):
        super().__init__()

        chars = ascii_lowercase + ascii_uppercase
        self.name = 'func_' + ''.join(sample(chars, 15))
        self.args = args
        self.kwargs = kwargs
        self.prompt = prompt
        self._static_context = static_context
        self.examples = Prompt(examples)
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.default = default

    def forward(self, *args, **kwargs) -> Expression:
        @few_shot(prompt=self.prompt,
                     examples=self.examples,
                     pre_processor=self.pre_processor,
                     post_processor=self.post_processor,
                     default=self.default,
                     *self.args, **self.kwargs)
        def _func(_):
            pass
        _type = type(self.name, (Expression, ), {
            # constructor
            "forward": _func,
            "_sym_return_type": self.name,
        })
        obj = _type()
        obj._sym_return_type = _type
        return self._to_symbol(obj(*args, **kwargs))


class SimilarityClassification(Expression):
    def __init__(self, classes: List[str], metric: str = 'cosine', in_memory: bool = False):
        super().__init__()
        self.classes   = classes
        self.metric    = metric
        self.in_memory = in_memory

        if self.in_memory:
            CustomUserWarning(f'Caching mode is enabled! It is your responsability to empty the .cache folder if you did changes to the classes. The cache is located at {Path.home()}/.symai/cache')

    def forward(self, x: Symbol) -> Symbol:
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


class Indexer(Expression):
    def __init__(self, index_name: str = 'data-index', top_k: int = 10, batch_size: int = 20):
        super().__init__()
        self.index_name = index_name
        self.elements   = []
        self.batch_size = batch_size
        self.top_k      = top_k
        self.NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

    def split_paragraphs(self, input_text=""):
        no_newlines = input_text.strip("\n")  # remove leading and trailing "\n"
        split_text = self.NEWLINES_RE.split(no_newlines)  # regex splitting

        paragraphs = [p + "\n" for p in split_text if p.strip()]
        # p + "\n" ensures that all lines in the paragraph end with a newline
        # p.strip() == True if paragraph has other characters than whitespace

        return paragraphs

    def split_huge_paragraphs(self, input_text: List[str], max_length=400):
        paragraphs = []
        for text in input_text:
            words = text.split()
            if len(words) > max_length:
                for i in range(0, len(words), max_length):
                    paragraph = ' '.join(words[i:i + max_length])
                    paragraphs.append(paragraph + "\n")
            else:
                paragraphs.append(text)
        return paragraphs

    def forward(self, query: Optional[Symbol] = None, *args, **kwargs) -> Symbol:
        that = self
        if query is not None:
            query = self._to_symbol(query)
            # split text paragraph-wise and index each paragraph separately
            self.elements = self.split_paragraphs(query.value)
            self.elements = self.split_huge_paragraphs(self.elements)
            # run over the elments in batches
            for i in tqdm(range(0, len(self.elements), self.batch_size)):
                val = Symbol(self.elements[i:i+self.batch_size]).zip()
                that.add(val)

        def _func(query):
            res = that.get(Symbol(query).embed().value, index_top_k=that.top_k).ast()
            res = [v['metadata']['text'] for v in res['matches']]
            sym = that._to_symbol(res)
            rsp = sym.query(query, max_tokens=2000)
            return rsp

        return _func


class DocumentRetriever(Expression):
    def __init__(self, file_path: str):
        super().__init__()
        reader = FileReader()
        indexer = Indexer()
        text = reader(file_path)
        self.index = indexer(text)

    def forward(self, query: Optional[Symbol]) -> Symbol:
        return self.index(query)
