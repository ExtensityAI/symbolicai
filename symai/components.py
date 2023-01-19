from typing import Callable, Iterator, List, Optional
from .symbol import Symbol, Expression


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


class Choice(Expression):
    def __init__(self, cases: List[str], default: Optional[str] = None):
        super().__init__()
        self.cases: List[str] = cases
        self.default: Optional[str] = default
    
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        return sym.choice(cases=self.cases, default=self.default, *args, **kwargs)


class Output(Expression):
    def __init__(self, expr: Expression, handler: Callable, verbose: bool = False):
        super().__init__()
        self.expr: Expression = expr
        self.handler: Callable = handler
        self.verbose: bool = verbose
        
    def forward(self, *args, **kwargs) -> Symbol:
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
    def __init__(self, expr: Expression, force: bool = False):
        super().__init__()
        self.max_tokens: int = 3000 #4097
        self.char_token_ratio: float = 0.6
        self.expr: Expression = expr
        self.force: bool = force
    
    def forward(self, sym: Symbol, **kwargs) -> Iterator[Symbol]:
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
    def __init__(self, expr: Expression):
        super().__init__()
        self.expr: Expression = expr
    
    def forward(self, *args, **kwargs) -> Symbol:
        self.command(verbose=True)
        res = self.expr(*args, **kwargs)
        self.command(verbose=False)
        return res
    
    
class Analyze(Expression):
    def __init__(self, exception: Exception, query: Optional[str] = None):
        super().__init__()
        self.exception: Expression = exception
        self.query: Optional[str] = query
    
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        return sym.analyze(exception=self.exception, query=self.query, *args, **kwargs)
    
    
class Log(Expression):
    def __init__(self, expr: Expression):
        super().__init__()
        self.expr: Expression = expr
    
    def forward(self, *args, **kwargs) -> Symbol:
        self.command(logging=True)
        res = self.expr(*args, **kwargs)
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
    def __init__(self, description: str, libraries: List[str] = []):
        super().__init__()
        self.description: str = description
        self.libraries: List[str] = libraries
        
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        return sym.style(description=self.description, libraries=self.libraries, **kwargs)

    
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
        return self.open(path, **kwargs)
