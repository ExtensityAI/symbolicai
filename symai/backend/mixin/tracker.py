import sys

from ... import prompts as prm
from ... import core_ext
from ..engines.embedding.engine_openai import EmbeddingEngine
from ..engines.neurosymbolic.engine_openai_gptX_chat import GPTXChatEngine
from ..mixin.openai import SUPPORTED_MODELS
from ...symbol import Symbol
from ...utils import CustomUserWarning


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
            prompt   = frame.f_locals.get('argument').prop.prompt
            examples = frame.f_locals.get('argument').prop.examples

            if prompt is not None:
                if isinstance(prompt, str): inp += prompt + '\n'

            if examples is not None:
                if    isinstance(examples, str): inp += examples
                elif  isinstance(examples, list): inp += '\n'.join(examples)
                elif  isinstance(examples, prm.Prompt): inp += examples.__repr__()

            self._inputs.append(len(Symbol(inp).tokens))

        elif isinstance(engine, EmbeddingEngine):
            if self._embedding_model() not in self._supported_models: return

            text = frame.f_locals.get('argument').prop.instance.value

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

    @core_ext.bind(engine='neurosymbolic', property='model')
    def _neurosymbolic_model(self): pass

    @core_ext.bind(engine='neurosymbolic', property='pricing')
    def _neurosymbolic_pricing(self): pass

    @core_ext.bind(engine='embedding', property='model')
    def _embedding_model(self): pass

    @core_ext.bind(engine='embedding', property='pricing')
    def _embedding_pricing(self): pass