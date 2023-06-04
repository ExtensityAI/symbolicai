from typing import Callable, Iterator, List, Optional
from .symbol import Symbol, Expression
import symai as ai
from random import sample
from string import ascii_uppercase, ascii_lowercase


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
        return sym.choice(cases=self.cases, default=self.default, *args, **kwargs)
    

class Classify(Expression):
    def __init__(self, options: List[str]):
        super().__init__()
        self.options: List[str] = options

    @ai.cache(in_memory=False)
    def embed_options(self):
        opts = map(Expression, self.options)
        embeddings = [opt.embed() for opt in opts]
        return embeddings
    
    def forward(self, message: Symbol, *args, **kwargs) -> Symbol:
        usr_embed = message.embed()
        similarities = [usr_embed.similarity(emb) for emb in self.embed_options()]
        similarities = sorted(zip(self.options, similarities), key=lambda x: x[1], reverse=True)
        return self._sym_return_type(similarities[0][0])


class Output(Expression):
    def __init__(self, expr: Expression, handler: Callable, verbose: bool = False):
        super().__init__()
        self.expr: Expression = expr
        self.handler: Callable = handler
        self.verbose: bool = verbose
        
    def forward(self, *args, **kwargs) -> Symbol:
        kwargs['verbose'] = self.verbose
        kwargs['handler'] = self.handler
        return Expression.output(expr=self.expr, *args, **kwargs)


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
    def __init__(self, expr: Expression, force: bool = False):
        super().__init__()
        self.max_tokens: int = 4000 #4097
        self.char_token_ratio: float = 0.6
        self.expr: Expression = expr
        self.force: bool = force
    
    def forward(self, sym: Expression, **kwargs) -> Iterator[Symbol]:
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
    
    def forward(self, *args, **kwargs) -> Symbol:
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
    
    def forward(self, *args, **kwargs) -> Symbol:
        Expression.command(logging=True, engines=self.engines)
        res = self.expr(*args, **kwargs)
        Expression.command(logging=False, engines=self.engines)
        return res


class Template(Expression):
    def __init__(self, template: str = "<html><body>{{placeholder}}</body></html>", placeholder: str = '{{placeholder}}'):
        super().__init__()
        self.placeholder = placeholder
        self.template_ = template
        
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        return sym.template(self.template_, self.placeholder, **kwargs)
    
    
class Style(Expression):
    def __init__(self, description: str, libraries: List[str] = []):
        super().__init__()
        self.description: str = description
        self.libraries: List[str] = libraries
        
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        return sym.style(description=self.description, libraries=self.libraries, **kwargs)
    
    
class Query(Expression):
    def __init__(self, prompt: str):
        super().__init__()
        self.prompt: str = prompt
        
    def forward(self, sym: Symbol, context: Symbol = None, *args, **kwargs) -> Symbol:
        return sym.query(prompt=self.prompt, context=context, **kwargs)

    
class Outline(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.outline(**kwargs)
    
    
class Clean(Expression):    
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.clean(**kwargs)
    
    
class Execute(Expression):    
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.execute(**kwargs)
    
    
class Convert(Expression):
    def __init__(self, format: str = 'Python'):
        super().__init__()
        self.format = format
    
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.convert(format=self.format, **kwargs)
    

class Embed(Expression):    
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.embed(**kwargs)
    

class Cluster(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.cluster(**kwargs)
    
    
class Compose(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.compose(**kwargs)
    

class Map(Expression):
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.map(**kwargs)
    
    
class Translate(Expression):
    def __init__(self, language: str = 'English'):
        super().__init__()
        self.language = language
    
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        if sym.isinstanceof(f'{self.language} text'):
            return sym
        return sym.translate(language=self.language, **kwargs)


class IncludeFilter(Expression):
    def __init__(self, include: str):
        super().__init__()
        self.include = include
    
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.filter(self.include, include=True, **kwargs)


class ExcludeFilter(Expression):
    def __init__(self, exclude: str):
        super().__init__()
        self.exclude = exclude
        
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        return sym.filter(self.exclude, include=False, **kwargs)


class Open(Expression):
    def forward(self, path: str, **kwargs) -> Symbol:
        return Expression.open(path, **kwargs)


class FileQuery(Expression):
    def __init__(self, path: str, filter: str):
        super().__init__()
        self.path = path
        file_open = Open()
        self.query_stream = Stream(Sequence(
            IncludeFilter(filter),
        ))
        self.file = file_open(path)
        
    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        res = Symbol(list(self.query_stream(self.file)))
        return res.query(prompt=sym, context=res, **kwargs)


class Function(Expression):
    def __init__(self, prompt: str, static_context: str = "",
                 examples: Optional[ai.Prompt] = [],
                 pre_processor: Optional[List[ai.PreProcessor]] = None,
                 post_processor: Optional[List[ai.PostProcessor]] = None,
                 default: Optional[object] = None, *args, **kwargs):
        super().__init__()

        chars = ascii_lowercase + ascii_uppercase
        self.name = 'type_' + ''.join(sample(chars, 15))
        self.args = args
        self.kwargs = kwargs
        self.prompt = prompt
        self._static_context = static_context
        self.examples = examples
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.default = default
        
    def forward(self, *args, **kwargs) -> Symbol:
        @ai.few_shot(prompt=self.prompt, 
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
        return obj(*args, **kwargs)

