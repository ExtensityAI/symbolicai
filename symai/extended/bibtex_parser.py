from .. import core
from ..pre_processors import PreProcessor
from ..symbol import Expression, Symbol
from ..post_processors import CodeExtractPostProcessor


BIB_DESCRIPTION = """[Description]
You take in a text with references to papers and return a list of biblatex entries.
The cite reference is always having the last name of the author and then separated by a colon the year of publication, and optionally if multiple papers are published in the same year, a letter is appended to the year of publication, i.e. Agarwal:20a, Agarwal:20b, etc.
Exclude URLs from the biblatex entries.

[Single Example]
>>>
An Optimistic Perspective on Offline Reinforcement Learning, [Submitted on 10 Jul 2019 (v1), last revised 22 Jun 2020 (this version, v4)] Rishabh Agarwal, Dale Schuurmans, Mohammad Norouzi, https://arxiv.org/abs/1907.04543, arXiv:1907.04543 [cs.LG]
<<<
```bibtex
@article{Agarwal:20,
      title={An Optimistic Perspective on Offline Reinforcement Learning},
      author={R. Agarwal and D. Schuurmans and M. Norouzi},
      year={2020},
      journal={arXiv preprint arXiv:1907.04543},
      howpublished={arXiv},
      primaryClass={cs.LG}
}
```

[Multiple Examples]
>>>
Benchmarking Batch Deep Reinforcement Learning Algorithms, [Submitted on 3 Oct 2019] Scott Fujimoto, Edoardo Conti, Mohammad Ghavamzadeh, Joelle Pineau 	arXiv:1910.01708 [cs.LG]

Off-Policy Deep Reinforcement Learning without Exploration, Scott Fujimoto, D. Meger, Doina Precup
Published in International Conference on… 7 December 2018 arXiv:1812.02900v3  [cs.LG]  10 Aug 2019

Multimodal Few-Shot Learning with Frozen Language Models Maria Tsimpoukelli
~Maria_Tsimpoukelli1
, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals, Felix Hill Published: 09 Nov 2021, Last Modified: 22 Oct 2023 NeurIPS 2021 Poster Advances in Neural Information Processing Systems 34 (NeurIPS 2021), pages 200--212

<<<
```bibtex
@article{Fujimoto:19a,
      title={Benchmarking Batch Deep Reinforcement Learning Algorithms},
      author={S. Fujimoto and E. Conti and M. Ghavamzadeh and J. Pineau},
      year={2019},
      journal={arXiv preprint arXiv:1910.01708},
      howpublished={arXiv},
      primaryClass={cs.LG}
}

@article{Fujimoto:19b,
      title={Off-Policy Deep Reinforcement Learning without Exploration},
      author={S. Fujimoto and D. Meger and D. Precup},
      year={2019},
      journal={arXiv preprint arXiv:1812.02900},
      howpublished={arXiv},
      primaryClass={cs.LG}
}

@inproceedings{tsimpoukelli:21,
	title = {Multimodal {Few}-{Shot} {Learning} with {Frozen} {Language} {Models}},
	booktitle = {Advances in {Neural} {Information} {Processing} {Systems} 34: {Annual} {Conference} on {Neural} {Information} {Processing} {Systems} 2021, {NeurIPS} 2021, {December} 6-14, 2021, virtual},
	author = {M. Tsimpoukelli and J. Menick and S. Cabi and S. M. A. Eslami and O. Vinyals and F. Hill},
	editor = {M.'A. Ranzato and A. Beygelzimer and Y. N. Dauphin and P. Liang and J. W. Vaughan},
	year = {2021},
	pages = {200--212},
}
```
"""


class BibTexPreProcessor(PreProcessor):
    def __call__(self, argument):
        return '>>>\n{}\n\n<<<\n'.format(str(argument.args[0]))


class BibTexParser(Expression):
    @property
    def static_context(self) -> str:
        return BIB_DESCRIPTION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sym_return_type = BibTexParser

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        @core.zero_shot(prompt="Create bibtex entries:\n",
                   pre_processors=[BibTexPreProcessor()],
                   post_processors=[CodeExtractPostProcessor()], **kwargs)
        def _func(_, text) -> str:
            pass
        return _func(self, sym)
