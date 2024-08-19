import inspect
import os
import re
from collections import defaultdict
from pathlib import Path
from random import sample
from string import ascii_lowercase, ascii_uppercase
from typing import Callable, Iterator, List, Optional, Type, Union

import numpy as np
from attr import dataclass
from box import Box
from pydantic import BaseModel, ValidationError
from pyvis.network import Network
from tqdm import tqdm

from . import core, core_ext
from .constraints import DictFormatConstraint
from .formatter import ParagraphFormatter
from .post_processors import (CodeExtractPostProcessor,
                              JsonTruncateMarkdownPostProcessor,
                              JsonTruncatePostProcessor, PostProcessor,
                              StripPostProcessor)
from .pre_processors import JsonPreProcessor, PreProcessor
from .processor import ProcessorPipeline
from .prompts import JsonPromptTemplate, Prompt
from .symbol import Expression, Metadata, Symbol
from .utils import CustomUserWarning
from .backend.settings import HOME_PATH


class GraphViz(Expression):
    def __init__(self,
                 notebook = True,
                 cdn_resources = "remote",
                 bgcolor = "#222222",
                 font_color = "white",
                 height = "750px",
                 width = "100%",
                 select_menu = True,
                 filter_menu = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.net  = Network(notebook=notebook,
                            cdn_resources=cdn_resources,
                            bgcolor=bgcolor,
                            font_color=font_color,
                            height=height,
                            width=width,
                            select_menu=select_menu,
                            filter_menu=filter_menu)

    def forward(self, sym: Symbol, file_path: str, **kwargs):
        nodes = [str(n) if n.value else n.__repr__(simplified=True) for n in sym.nodes]
        edges = [(str(e[0]) if e[0].value else e[0].__repr__(simplified=True),
                  str(e[1]) if e[1].value else e[1].__repr__(simplified=True)) for e in sym.edges]
        self.net.add_nodes(nodes)
        self.net.add_edges(edges)
        file_path = file_path if file_path.endswith('.html') else file_path + '.html'
        return self.net.show(file_path)


class TrackerTraceable(Expression):
    pass


class Any(Expression):
    def __init__(self, *expr: List[Expression], **kwargs):
        super().__init__(**kwargs)
        self.expr: List[Expression] = expr

    def forward(self, *args, **kwargs) -> Symbol:
        return self.sym_return_type(any([e() for e in self.expr(*args, **kwargs)]))


class All(Expression):
    def __init__(self, *expr: List[Expression], **kwargs):
        super().__init__(**kwargs)
        self.expr: List[Expression] = expr

    def forward(self, *args, **kwargs) -> Symbol:
        return self.sym_return_type(all([e() for e in self.expr(*args, **kwargs)]))


class Try(Expression):
    def __init__(self, expr: Expression, retries: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.expr: Expression = expr
        self.retries: int = retries

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.ftry(self.expr, retries=self.retries, **kwargs)


class Lambda(Expression):
    def __init__(self, callable: Callable, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, cases: List[str], default: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.cases: List[str] = cases
        self.default: Optional[str] = default

    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.choice(cases=self.cases, default=self.default, *args, **kwargs)


class Output(Expression):
    def __init__(self, expr: Expression, handler: Callable, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.expr: Expression = expr
        self.handler: Callable = handler
        self.verbose: bool = verbose

    def forward(self, *args, **kwargs) -> Expression:
        kwargs['verbose'] = self.verbose
        kwargs['handler'] = self.handler
        return self.output(expr=self.expr, *args, **kwargs)


class Sequence(TrackerTraceable):
    def __init__(self, *expressions: List[Expression], **kwargs):
        super().__init__(**kwargs)
        self.expressions: List[Expression] = expressions

    def forward(self, *args, **kwargs) -> Symbol:
        sym = self.expressions[0](*args, **kwargs)
        metadata = Metadata()
        metadata.results = []
        metadata.results.append(sym)
        for expr in self.expressions[1:]:
            sym = expr(sym, **kwargs)
            metadata.results.append(sym)
        sym = self._to_symbol(sym)
        sym._metadata.results = metadata.results
        return sym


class Parallel(Expression):
    def __init__(self, *expr: List[Expression], sequential: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.sequential: bool       = sequential
        self.expr: List[Expression] = expr
        self.results: List[Symbol]  = []

    def forward(self, *args, **kwargs) -> Symbol:
        # run in sequence
        if self.sequential:
            return [e(*args, **kwargs) for e in self.expr]
        # run in parallel
        @core_ext.parallel(self.expr)
        def _func(e, *args, **kwargs):
            return e(*args, **kwargs)
        self.results = _func(*args, **kwargs)
        # final result of the parallel execution
        return self._to_symbol(self.results)


#@TODO: BinPacker(format="...") -> ensure that data packages form a "bin" that's consistent (e.g. never break a sentence in the middle)
class Stream(Expression):
    def __init__(self, expr: Optional[Expression] = None, retrieval: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, expr: Optional[Expression] = None, engines=['all'], **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, exception: Exception, query: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.exception: Expression = exception
        self.query: Optional[str] = query

    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        return sym.analyze(exception=self.exception, query=self.query, *args, **kwargs)


class Log(Expression):
    def __init__(self, expr: Optional[Expression] = None, engines=['all'], **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, template: str = "<html><body>{{placeholder}}</body></html>", placeholder: str = '{{placeholder}}', **kwargs):
        super().__init__(**kwargs)
        self.placeholder = placeholder
        self.template_ = template

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.template(template=self.template_, placeholder=self.placeholder, **kwargs)


class Extract(Expression):
    def __init__(self, query, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = query

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.extract(self.query, **kwargs)


class RuntimeExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # define function that executes expressions
        self.runner = Execute(enclosure=True)

    def forward(self, code: Symbol):
        code = self._to_symbol(code)
        # declare the runtime expression from the code
        expr = self.runner(code)
        def _func(sym):
            # execute nested expression
            return expr['locals']['_output_'](sym)
        return _func


class Metric(Expression):
    def __init__(self, normalize: bool = False, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, description: str, libraries: List[str] = [], **kwargs):
        super().__init__(**kwargs)
        self.description: str = description
        self.libraries: List[str] = libraries

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.style(description=self.description, libraries=self.libraries, **kwargs)


class Query(TrackerTraceable):
    def __init__(self, prompt: str, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, enclosure: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.enclosure = enclosure
        self.template = """# -*- code execution template -*-
def _func(*args, **kwargs):
    _value_obj_ = None
{sym}
    # assume that code assigns to _value_obj_ variable
    return _value_obj_
_output_ = _func()
"""

    def forward(self, sym: Symbol, enclosure: bool = False, **kwargs) -> Symbol:
        if enclosure or self.enclosure:
            lines = str(sym).split('\n')
            lines = ['    ' + line for line in lines]
            sym = '\n'.join(lines)
            sym = self.template.replace('{sym}', str(sym))
        sym = self._to_symbol(sym)
        return sym.execute(**kwargs)


class Convert(Expression):
    def __init__(self, format: str = 'Python', **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, language: str = 'English', **kwargs):
        super().__init__(**kwargs)
        self.language = language

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        if sym.isinstanceof(f'{self.language} text'):
            return sym
        return sym.translate(language=self.language, **kwargs)


class IncludeFilter(Expression):
    def __init__(self, include: str, **kwargs):
        super().__init__(**kwargs)
        self.include = include

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.filter(self.include, include=True, **kwargs)


class ExcludeFilter(Expression):
    def __init__(self, exclude: str, **kwargs):
        super().__init__(**kwargs)
        self.exclude = exclude

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return sym.filter(self.exclude, include=False, **kwargs)


class FileWriter(Expression):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        with open(self.path, 'w') as f:
            f.write(str(sym))


class FileReader(Expression):
    @staticmethod
    def exists(path: str) -> bool:
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

    @staticmethod
    def extract_files(cmds: str) -> List[str]:
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

    @staticmethod
    def expand_user_path(path: str) -> str:
        return path.replace('~', os.path.expanduser('~'))

    @staticmethod
    def integrity_check(files: List[str]) -> List[str]:
        not_skipped = []
        for file in tqdm(files):
            if FileReader.exists(file):
                not_skipped.append(file)
            else:
                CustomUserWarning(f'Skipping file: {file}')
        return not_skipped

    def forward(self, files: Union[str, List[str]], **kwargs) -> Expression:
        if isinstance(files, str):
            return self.open(files, **kwargs)
        if isinstance(files, list):
            if kwargs.get('run_integrity_check'):
                files = self.integrity_check(files)
            return self.sym_return_type([self.open(f, **kwargs).value for f in files])
        raise ValueError('Invalid input type. Please provide a string (file path) or a list of strings (file paths).')

class FileQuery(Expression):
    def __init__(self, path: str, filter: str, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, prompt: str       = '',
                 examples: Optional[str] = [],
                 pre_processors: Optional[List[PreProcessor]]   = None,
                 post_processors: Optional[List[PostProcessor]] = None,
                 default: Optional[object]       = None,
                 constraints: List[Callable]     = [],
                 return_type: Optional[Type]     = str,
                 sym_return_type: Optional[Type] = Symbol,
                 origin_type: Optional[Type]     = Expression,
                 *args, **kwargs):
        super().__init__(**kwargs)
        chars       = ascii_lowercase + ascii_uppercase
        self.name   = 'func_' + ''.join(sample(chars, 15))
        self.args   = args
        self.kwargs = kwargs
        self._promptTemplate     = prompt
        self._promptFormatArgs   = []
        self._promptFormatKwargs = {}
        self.examples        = Prompt(examples)
        self.pre_processors  = pre_processors
        self.post_processors = post_processors
        self.constraints     = constraints
        self.default         = default
        self.return_type     = return_type
        self.sym_return_type = sym_return_type
        self.origin_type     = origin_type

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
        _type = type(self.name, (self.origin_type, ), {
            # constructor
            "forward": _func,
            "sym_return_type": self.sym_return_type,
            "static_context": self.static_context,
            "dynamic_context": self.dynamic_context,
            "__class__": self.__class__,
            "__module__": self.__module__,
        })
        obj = _type()

        return self._to_symbol(obj(*args, **kwargs))


class PrepareData(Function):
    class PrepareDataPreProcessor(PreProcessor):
        def __call__(self, argument):
            assert argument.prop.context is not None
            instruct = argument.prop.prompt
            context  = argument.prop.context
            return """{
    'context': '%s',
    'instruction': '%s',
    'result': 'TODO: Replace this with the expected result.'
}""" % (context, instruct)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_processors  = [self.PrepareDataPreProcessor()]
        self.constraints     = [DictFormatConstraint({ 'result': '<the data>' })]
        self.post_processors = [JsonTruncateMarkdownPostProcessor()]
        self.return_type     = dict # constraint to cast the result to a dict

    @property
    def static_context(self):
        return """[CONTEXT]
Your goal is to prepare the data for the next task instruction. The data should follow the format of the task description based on the given context. Replace the `TODO` section with the expected result of the data preparation. Only provide the 'result' json-key as follows: ```json\n{ 'result': 'TODO:...' }\n```

[GENERAL TEMPLATE]
```json
{
    'context': 'The general context of the task.',
    'instruction': 'The next instruction of the task for the data preparation.',
    'result': 'The expected result of the data preparation.'
}
```

[EXAMPLE]
[Instruction]:
{
    'context': 'Write a document about the history of AI and include references to the following people: Alan Turing, John McCarthy, Marvin Minsky, and Yoshua Bengio.',
    'instruction': 'Google the history of AI for Alan Turing',
    'result': 'TODO'
}

[Result]:
```json
{
    'result': 'Alan Turing history of AI'
}
```
"""


class ExpressionBuilder(Function):
    def __init__(self, **kwargs):
        super().__init__('Generate the code following the instructions:', **kwargs)
        self.processors = ProcessorPipeline([StripPostProcessor(), CodeExtractPostProcessor()])

    def forward(self, instruct, *args, **kwargs):
        result = super().forward(instruct)
        return self.processors(str(result), None)

    @property
    def static_context(self):
        return """[Description]
Your goal is to generate the code of the forward function following the instruction of the extracted task and task description. Expect that all imports are already defined. Only produce the code for the TODO section as shown below:

[Template]
```python
# do not remove or change the imports
from symai import Function, Expression, Symbol
class TODOExpression(Expression):
    # initialize the expression with task specific arguments
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        # TODO: Place here the generated code.

    # define the forward function with data specific arguments
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym) # ensure that sym is a Symbol
        result = None
        # TODO: Place here the generated code.
        return result
# assign the expression type to the variable _value_obj_
_value_obj_ = TODOExpression
```

[API and Capabilities]
Function: Define a function with a task description.
func = Function('task description') # example: Function('Extract city names from the weather report.')
res  = func('data') # example: func('The weather in New York is 20 degrees.')
res # str containing the result of the function i.e. 'New York'
Symbol: Define a symbol object that can be used to hold data.
sym  = Symbol('data') # example: Symbol('The weather in New York is 20 degrees.')
sym.value # value type containing the data i.e. 'The weather in New York is 20 degrees.'
Expression: Define a custom expression that can be used to define a new task.
expr = Expression() # example: Expression() that has a __init__ and forward function.
res  = expr(sym) # example: expr(sym) where sym is a Symbol object.

Always produce the entire code to be executed in the same Python process. All task expressions and functions are defined in the same process. Generate the code within the ```python # TODO: ``` code block as shown in the example. The code must be self-contained, include all imports and executable. Do NOT remove or change the template code or the imports. Only replace the parts with TODO in the respective section and expression name with your generated code.
"""


class JsonParser(Expression):
    def __init__(self, query: str, json_: dict, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, classes: List[str], metric: str = 'cosine', in_memory: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.classes   = classes
        self.metric    = metric
        self.in_memory = in_memory

        if self.in_memory:
            CustomUserWarning(f'Caching mode is enabled! It is your responsability to empty the .cache folder if you did changes to the classes. The cache is located at {HOME_PATH}/.symai/cache')

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
    def __init__(self, blueprint: Prompt, **kwargs):
        super().__init__(**kwargs)
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

    def __init__(
            self,
            index_name: str = DEFAULT,
            top_k: int = 8,
            batch_size: int = 20,
            formatter: Callable = ParagraphFormatter(),
            auto_add=False,
            raw_result: bool = False,
            new_dim: int = 1536,
            **kwargs
        ):
        super().__init__(**kwargs)
        index_name = Indexer.replace_special_chars(index_name)
        self.index_name = index_name
        self.elements   = []
        self.batch_size = batch_size
        self.top_k      = top_k
        self.retrieval  = None
        self.formatter  = formatter
        self.raw_result = raw_result
        self.new_dim    = new_dim
        self.sym_return_type = Expression

        # append index name to indices.txt in home directory .symai folder (default)
        self.path = HOME_PATH / '.symai' / 'indices.txt'
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
        path = HOME_PATH / '.symai' / 'indices.txt'
        if not path.exists():
            return False
        with open(path, 'r') as f:
            indices = f.read().split('\n')
            if self.index_name in indices:
                return True

    def forward(
            self,
            data: Optional[Symbol] = None,
            raw_result: bool = False,
        ) -> Symbol:
        that = self
        if data is not None:
            data = self._to_symbol(data)
            self.elements = self.formatter(data).value
            # run over the elments in batches
            for i in tqdm(range(0, len(self.elements), self.batch_size)):
                val = Symbol(self.elements[i:i+self.batch_size]).zip(new_dim=self.new_dim)
                that.add(val, index_name=that.index_name, index_dims=that.new_dim)
            that.config(None, index_name=that.index_name, index_dims=that.new_dim)

        def _func(query, *args, **kwargs):
            raw_result = kwargs.get('raw_result') or that.raw_result
            query_emb = Symbol(query).embed(new_dim=that.new_dim).value
            res = that.get(query_emb, index_name=that.index_name, index_top_k=that.top_k, ori_query=query, index_dims=that.new_dim, **kwargs)
            that.retrieval = res
            if raw_result:
                return res
            rsp = res.query(query, *args, **kwargs)
            return rsp

        return _func


class PrimitiveDisabler(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._primitives = set()
        self._original_primitives = defaultdict(list)

    def __enter__(self):
        # Avoid circular imports; import locally
        from .symbol import Symbol

        frame = inspect.currentframe()
        f_locals = frame.f_back.f_locals
        self._symbols = {key: value for key, value in f_locals.items() if isinstance(value, Symbol)}
        self._extract_primitives()
        self._disable_primitives()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._enable_primitives()

    def _disable_primitives(self):
        for sym_name, sym in self._symbols.items():
            for func in self._primitives:
                if hasattr(sym, func):
                    self._original_primitives[sym_name].append((func, getattr(sym, func)))
                    setattr(sym, func, lambda *args, **kwargs: None)

    def _enable_primitives(self):
        for sym_name, sym in self._symbols.items():
            for func, value in self._original_primitives[sym_name]:
                setattr(sym, func, value)

    def _extract_primitives(self):
        for sym in self._symbols.values():
            for primitive in sym._primitives:
                for method, _ in inspect.getmembers(primitive, predicate=inspect.isfunction):
                    if method in self._primitives or method.startswith('_'):
                        continue
                    self._primitives.add(method)


class FunctionWithUsage(Function):
    def __init__(
        self,
        missing_usage_exception: bool = True,
        verbose=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.missing_usage_exception = missing_usage_exception
        self.verbose = verbose

    def print_verbose(self, msg):
        if self.verbose:
            print(msg)

    def _format_usage(self, prompt_tokens, completion_tokens, total_tokens):
        return Box(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )

    def reset_usage(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def add_usage(self, usage):
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens

    def get_usage(self):
        return self._format_usage(
            self.prompt_tokens, self.completion_tokens, self.total_tokens
        )

    def forward(self, *args, **kwargs):
        if "return_metadata" not in kwargs:
            kwargs["return_metadata"] = True

        res, metadata = super().forward(*args, **kwargs)

        raw_output = metadata.get("raw_output")
        if hasattr(raw_output, "usage"):
            usage = raw_output.usage
            prompt_tokens = (
                usage.prompt_tokens if hasattr(usage, "prompt_tokens") else 0
            )
            completion_tokens = (
                usage.completion_tokens if hasattr(usage, "completion_tokens") else 0
            )
            total_tokens = usage.total_tokens if hasattr(usage, "total_tokens") else 0

            self.print_verbose(
                f"[Usage] Prompt: {prompt_tokens} Completion: {completion_tokens} Total: {total_tokens}"
            )

            # keep running total
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
        else:
            if self.missing_usage_exception and not "preview" in kwargs:
                raise Exception("Missing usage in metadata of neursymbolic engine")
            else:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

        return res, self._format_usage(prompt_tokens, completion_tokens, total_tokens)


class ValidatedFunction(FunctionWithUsage):
    def __init__(
        self,
        data_model: BaseModel = None,
        retry_count=5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retry_count = retry_count
        self.data_model = data_model

    def prepare_seeds(self, num_seeds: int, **kwargs):
        # get list of seeds for remedy (to avoid same remedy for same input)
        if "seed" in kwargs:
            seed = kwargs["seed"]
        elif hasattr(self, "seed"):
            seed = self.seed
        else:
            seed = 42

        rnd = np.random.RandomState(seed=seed)
        seeds = rnd.randint(
            0, np.iinfo(np.int16).max, size=num_seeds, dtype=np.int16
        ).tolist()
        return seeds

    def forward(self, *args, **kwargs):
        # force JSON mode
        kwargs["response_format"] = {"type": "json_object"}
        if "JSON" not in self.static_context:
            raise Exception("The static context must contain the string 'JSON'")

        json, usage = super().forward(*args, **kwargs)

        # get list of seeds for remedy (to avoid same remedy for same input)
        remedy_seeds = self.prepare_seeds(self.retry_count, **kwargs)

        # prepare remedy function
        remedy_function = FunctionWithUsage(
            (
                "The following string should be in JSON format but contains errors. "
                + "Fix the errors and return a valid JSON string that is valid for the given JSON schema without changing values.\n"
            ),
            static_context="You are an agent for validating JSON schemas and fixing errors.",
            response_format={"type": "json_object"},
        )

        # try to validate the JSON and fix if necessary
        result = None
        for i in range(self.retry_count):
            try:
                # try to validate against provided data model
                result = self.data_model.model_validate_json(json, strict=True)
                break
            except ValidationError as e:
                # in case of validation error collect and format error messages
                error_str = ""
                for error in e.errors():
                    error_str += f"[ERROR] {error['msg']}: {error['loc']}\n"

                # ask model to fix the error
                self.print_verbose(f"[Retry {i + 1}/{self.retry_count}] ValidationError: {str(e)}")
                remedy_function.clear()
                remedy_function.adapt(
                    "[[JSON Schema]]\n"
                    + str(self.data_model.model_json_schema(mode="validation"))
                    + "\n\n"
                )
                remedy_function.adapt("[[ERRORS]]\n" + error_str + "\n")
                json, remedy_usage = remedy_function(json, seed=remedy_seeds[i])

                # update usage
                usage.prompt_tokens += remedy_usage.prompt_tokens
                usage.completion_tokens += remedy_usage.completion_tokens
                usage.total_tokens += remedy_usage.total_tokens

                # update global usage
                self.add_usage(remedy_usage)

        if result is None:
            raise Exception("Failed to retrieve valid JSON")

        return result, usage


@dataclass
class LengthConstraint:
    field_name: str
    min_characters: int = 0
    max_characters: int = 1000


class LengthConstrainedFunction(ValidatedFunction):
    def __init__(
        self,
        character_constraints: List[LengthConstraint] | LengthConstraint,
        constraint_retry_count: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.character_constraints = character_constraints
        self.constraint_retry_count = constraint_retry_count

    def forward(self, *args, **kwargs):
        result, usage = super().forward(*args, **kwargs)

        original_task = args[0]  # TODO can we validate this?

        # get list of seeds for remedy (to avoid same remedy for same input)
        remedy_seeds = self.prepare_seeds(self.constraint_retry_count, **kwargs)

        for i in range(self.constraint_retry_count):
            constraint_violations = self.check_constraints(result)
            if len(constraint_violations) > 0:
                self.print_verbose(f"Constraint violations: {constraint_violations}")
                remedy_task = self.wrap_task(
                    original_task, result.model_dump_json(), constraint_violations
                )
                kwargs["seed"] = remedy_seeds[i]
                result, remedy_usage = super().forward(remedy_task, *args[1:], **kwargs)
                # update local usage
                usage.prompt_tokens += remedy_usage.prompt_tokens
                usage.completion_tokens += remedy_usage.completion_tokens
                usage.total_tokens += remedy_usage.total_tokens
            else:
                break

        if i == self.constraint_retry_count and len(self.check_constraints(result)) > 0:
            raise Exception("Failed to enforce constraints")

        return result, usage

    def check_constraints(self, result: BaseModel):
        constraint_violations = []
        if type(self.character_constraints) is not list:
            character_constraints = [self.character_constraints]
        else:
            character_constraints = self.character_constraints

        for constraint in character_constraints:
            for field_value in self.get(result, constraint.field_name):
                if (
                    not constraint.min_characters
                    <= len(field_value)
                    <= constraint.max_characters
                ):
                    # TODO improve for lists (especially table of contents) by providing the index
                    self.print_verbose(
                        f"Field {constraint.field_name} must have between {constraint.min_characters} and {constraint.max_characters} characters, has {len(field_value)}"
                    )
                    remedy_str = [
                        f"The field {constraint.field_name} must have between {constraint.min_characters} and {constraint.max_characters} characters, but has {len(field_value)}."
                    ]
                    if len(field_value) < constraint.min_characters:
                        remedy_str.append(
                            f"Increase the length of {constraint.field_name} by at least {constraint.min_characters - len(field_value)} characters."
                        )
                    elif len(field_value) > constraint.max_characters:
                        remedy_str.append(
                            f"Decrease the length of {constraint.field_name} by at least {len(field_value) - constraint.max_characters} characters."
                        )
                    constraint_violations.append(" ".join(remedy_str))
                else:
                    self.print_verbose(
                        f"[PASS] Field {constraint.field_name} passed length validation: {len(field_value)} ({constraint.min_characters} - {constraint.max_characters})"
                    )

        return constraint_violations

    def wrap_task(self, task: str, result: str, violations: List[str]):
        wrapped_task = [
            "Your task was the following: \n\n" + task + "\n",
            "Your output was the following: \n\n" + result + "\n",
            "However, the output did violate the following constraints:",
        ]
        for constraint_violation in violations:
            wrapped_task.append(constraint_violation)
        wrapped_task.append(
            "\nFollow the original task and make sure to adhere to the constraints."
        )
        return "\n".join(wrapped_task)

    @staticmethod
    def get(obj, path: str):
        value = obj
        for i, key in enumerate(path.split(".")):
            if isinstance(value, list):
                try:
                    index = int(key)
                    if index < len(value):
                        value = value[index]
                    else:
                        raise ValueError(f"Index {index} out of range in {value}")
                except:
                    values = []
                    for val in value:
                        leaf = LengthConstrainedFunction.get(
                            val, ".".join(path.split(".")[i:])
                        )
                        values.extend(leaf)
                    return values
            elif isinstance(value, dict):
                if key in value:
                    value = value[key]
                else:
                    raise ValueError(f"Key {key} not found in {value}")
            else:
                if hasattr(value, key):
                    value = getattr(value, key)
                else:
                    raise ValueError(f"Key {key} not found in {value}")

        if not isinstance(value, list):
            value = [value]
        return value

