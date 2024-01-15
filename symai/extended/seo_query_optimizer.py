from typing import Callable

from .. import core
from ..formatter import SentenceFormatter
from ..post_processors import StripPostProcessor
from ..pre_processors import PreProcessor
from ..prompts import Prompt
from ..symbol import Expression, Symbol


SEO_OPTIMIZER_DESCRIPTION = """[Description]
You are a SEO query optimizer. You are given a list of queries, phrases or sentences and you need to optimize them for search engines.
Assume your search engines are based on vector databases and contain indices of GitHub repositories, papers and other resources.
To retrieve the information cosine similarity is used between semantic embeddings.
To optimize the queries, you need to extract the most information such as relevant entities and relationships between them from the [DATA] and [PAYLOAD] section.
Try to also extract also closely related entities and relationships based on what the user might be looking for, given the provided context.
The number of resulting queries should be between 1 and 8 statements separated by a comma.
"""


class SEOQueryOptimizerPreProcessor(PreProcessor):
    def __call__(self, argument):
        return '$> {} =>'.format(str(argument.args[0]))


class SEOQueryOptimizer(Expression):
    @property
    def static_context(self) -> str:
        return SEO_OPTIMIZER_DESCRIPTION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sym_return_type = SEOQueryOptimizer

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        @core.few_shot(prompt="Extract relationships between entities:\n",
                  examples=Prompt([
                        '$> John has a dog. =>John dog EOF',
                        '$> How can i find on wikipedia an article about programming? Preferably about python programming. =>Wikipedia python programming tutorial EOF',
                        '$> Similarly, the term general linguistics is used to distinguish core linguistics from other types of study =>general linguistics term, core linguistics from other types of study EOF',
                  ]),
                  pre_processors=[SEOQueryOptimizerPreProcessor()],
                  post_processors=[StripPostProcessor()],
                  stop=['EOF'], **kwargs)
        def _func(_, text) -> str:
            pass

        return _func(self, sym)


