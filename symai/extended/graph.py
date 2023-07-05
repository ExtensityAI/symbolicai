from multiprocessing import Pool

from ..core import *
from ..formatter import SentenceFormatter
from ..post_processors import StripPostProcessor
from ..pre_processors import PreProcessor
from ..prompts import Prompt
from ..symbol import Expression, Symbol

GRAPH_DESCRIPTION = """[Description]
Build source-target relationship pairs for named entities based for the [DATA] section. The [DATA] section contains one sentence.
Format all extracted pairs in a CSV format with the following columns: source, target, count.
The source is related to the target based on intermediate terms, such as "has a", "has two", "is used to" etc.
If more than one entity pair is extracted from the same sentence, then the CSV file should contain multiple rows separated by a newline (\\n)
"""


class GraphPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args, **kwds):
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return '$> {} =>'.format(str(args[0]))


class Graph(Expression):
    @property
    def static_context(self) -> str:
        return GRAPH_DESCRIPTION

    @property
    def _sym_return_type(self):
        return Graph

    def __init__(self, formatter: Callable = SentenceFormatter(), n_workers: int = 4):
        super().__init__()
        self.formatter = formatter
        self.n_workers = n_workers

    def process_symbol(self, s, *args, **kwargs):
        res = ''

        @few_shot(prompt="Extract relationships between entities:\n",
                  examples=Prompt([
                        '$> John has a dog. =>John, dog, 1 EOF',
                        '$> Karl has two sons. =>Karl, sons, 2 EOF',
                        '$> Similarly, the term general linguistics is used to distinguish core linguistics from other types of study =>general linguistics, core linguistics, 1 EOF',
                        '$> X has Y and Z has Y =>X, Y, 1\nZ, Y, 1 EOF',
                  ]),
                  pre_processor=[GraphPreProcessor()],
                  post_processor=[StripPostProcessor()],
                  stop=['EOF'], **kwargs)
        def _func(_, text) -> str:
            pass

        if len(str(s)) > 0:
            r = _func(self, s)
            rec = str(r)
            lines = rec.split('\n')
            for l in lines:
                l = l.strip()
                if len(l) > 0:
                    csv = l.split(',')
                    try:
                        if len(csv) == 3 and csv[0].strip() != '' and csv[1].strip() != '' and csv[2].strip() > 0:
                            test_ = int(csv[-1])
                            res += l + '\n'
                    except:
                        pass
        return res

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        res = 'source,target,value\n'
        sym_list = self.formatter(sym).value
        with Pool(self.n_workers) as p:
            results = p.map(self.process_symbol, sym_list)
        for r in results:
            res += r
        return res
