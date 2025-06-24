import functools
from typing import Any, Callable, Dict, List, Optional, Type

from box import Box

from . import post_processors as post
from . import pre_processors as pre
from . import prompts as prm
from .functional import EngineRepository
from .symbol import Expression, Metadata


class Argument(Expression):
    _default_suppress_verbose_output            = False
    _default_parse_system_instructions          = False
    _default_preview_value                      = False

    def __init__(self, args, signature_kwargs, decorator_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.args             = args # there is only signature args
        self.signature_kwargs = signature_kwargs.copy()
        self.decorator_kwargs = decorator_kwargs.copy()
        self.kwargs           = self._construct_kwargs(signature_kwargs=signature_kwargs,
                                                       decorator_kwargs=decorator_kwargs)
        self.prop             = Metadata()
        self._set_all_kwargs_as_properties()
        # Set default values if not specified for backend processing
        # Reserved keywords
        if 'preview' not in self.kwargs: # used for previewing the input (also for operators)
            self.prop.preview = Argument._default_preview_value
        if 'raw_input' not in self.kwargs:
            self.prop.raw_input = False
        if 'raw_output' not in self.kwargs:
            self.prop.raw_output = False
        if 'return_metadata' not in self.kwargs:
            self.prop.return_metadata = False
        if 'logging' not in self.kwargs:
            self.prop.logging = False
        if 'verbose' not in self.kwargs:
            self.prop.verbose = False
        if 'self_prompt' not in self.kwargs:
            self.prop.self_prompt = False
        if 'truncation_percentage' not in self.kwargs:
            self.prop.truncation_percentage = None
        if 'truncation_type' not in self.kwargs:
            self.prop.truncation_type = 'tail'
        if 'response_format' not in self.kwargs:
            self.prop.response_format = None
        if 'log_level' not in self.kwargs:
            self.prop.log_level = None
        if 'time_clock' not in self.kwargs:
            self.prop.time_clock = None
        if 'payload' not in self.kwargs:
            self.prop.payload = None
        if 'processed_input' not in self.kwargs:
            self.prop.processed_input = None
        if 'template_suffix' not in self.kwargs:
            self.prop.template_suffix = None
        if 'input_handler' not in self.kwargs:
            self.prop.input_handler = None
        if 'output_handler' not in self.kwargs:
            self.prop.output_handler = None
        if 'suppress_verbose_output' not in self.kwargs:
            self.prop.suppress_verbose_output = Argument._default_suppress_verbose_output
        if 'parse_system_instructions' not in self.kwargs:
            self.prop.parse_system_instructions = Argument._default_parse_system_instructions

    def _set_all_kwargs_as_properties(self):
        for key, value in self.kwargs.items():
            setattr(self.prop, key, value)

    @property
    def value(self):
        return Box({
            'args': self.args,
            'signature_kwargs': self.signature_kwargs,
            'decorator_kwargs': self.decorator_kwargs,
            'kwargs': self.kwargs,
            'prop': self.prop
        })

    def _construct_kwargs(self, signature_kwargs, decorator_kwargs):
        '''
        Combines and overrides the decorator args and kwargs with the runtime signature args and kwargs.

        Args:
            signature_kwargs (Dict): The signature kwargs.
            decorator_kwargs (Dict): The decorator kwargs.

        Returns:
            Dict: The combined and overridden kwargs.
        '''
        # Initialize with the decorator kwargs
        kwargs = {**decorator_kwargs}
        # Override the decorator kwargs with the signature kwargs
        for key, value in signature_kwargs.items():
            kwargs[key] = value
        return kwargs


def few_shot(prompt: str = '',
             examples: prm.Prompt = [],
             constraints: List[Callable] = [],
             default: Any = None,
             limit: int = 1,
             pre_processors: Optional[List[pre.PreProcessor]] = None,
             post_processors: Optional[List[post.PostProcessor]] = None,
             **decorator_kwargs):
    """"General decorator for the neural processing engine."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['prompt']   = prompt
            decorator_kwargs['examples'] = examples
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='neurosymbolic',
                                instance=instance,
                                func=func,
                                constraints=constraints,
                                default=default,
                                limit=limit,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def zero_shot(prompt: str = '',
              constraints: List[Callable] = [],
              default: Optional[object] = None,
              limit: int = 1,
              pre_processors: Optional[List[pre.PreProcessor]] = None,
              post_processors: Optional[List[post.PostProcessor]] = None,
              **decorator_kwargs):
    """"General decorator for the neural processing engine."""
    return few_shot(prompt,
                    examples=[],
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def prompt(message: str,
           **decorator_kwargs):
    """General decorator for the neural processing engine."""
    return few_shot(processed_input=message,
                    raw_input=True,
                    **decorator_kwargs)


def summarize(prompt: str = 'Summarize the content of the following text:\n',
              context: Optional[str] = None,
              constraints: List[Callable] = [],
              default: Optional[object] = None,
              limit: int = 1,
              stop: str | None = None,
              pre_processors: Optional[List[pre.PreProcessor]] = [pre.SummaryPreProcessing()],
              post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
              **decorator_kwargs):
    """Summarizes the content of a text."""
    return few_shot(prompt,
                    context=context,
                    examples=[],
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def equals(context: str = 'contextually',
           default: bool = False,
           prompt: str = "Make a fuzzy equals comparison; are the following objects {} the same?\n",
           examples: prm.Prompt = prm.FuzzyEquals(),
           constraints: List[Callable] = [],
           limit: int = 1,
           stop: str | None = None,
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.EqualsPreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """Equality function for two objects."""
    return few_shot(prompt=prompt.format(context),
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    max_tokens=None,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def sufficient(query: str,
               prompt: str = "Consider if there is sufficient information to answer the query:\n",
               default: bool = False,
               examples: prm.Prompt = prm.SufficientInformation(),
               constraints: List[Callable] = [],
               limit: int = 1,
               stop: str | None = None,
               pre_processors: Optional[List[pre.PreProcessor]] = [pre.SufficientInformationPreProcessor()],
               post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
               **decorator_kwargs) -> bool:
    """Determines if there is sufficient information to answer the given query."""
    return few_shot(prompt=prompt,
                    query=query,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    max_tokens=None,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def delitem(default: Optional[str] = None,
            prompt: str = "Delete the items at the index position:\n",
            examples: prm.Prompt = prm.RemoveIndex(),
            constraints: List[Callable] = [],
            limit: int = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.DeleteIndexPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Deletes the items at the specified index position."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def setitem(default: Optional[str] = None,
            prompt: str = "Set item at index position:\n",
            examples: prm.Prompt = prm.SetIndex(),
            constraints: List[Callable] = [],
            limit: int = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.SetIndexPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Sets an item at a given index position in a sequence."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def getitem(default: Optional[str] = None,
            prompt: str = "Get item at index position:\n",
            examples: prm.Prompt = prm.Index(),
            constraints: List[Callable] = [],
            limit: int = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.IndexPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Retrieves the item at the given index position."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def modify(changes: str,
           default: Optional[str] = None,
           prompt: str = "Modify the text to match the criteria:\n",
           examples: prm.Prompt = prm.Modify(),
           constraints: List[Callable] = [],
           limit: int = 1,
           stop: str | None = None,
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.ModifyPreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """A function to modify a text based on a set of criteria."""
    return few_shot(prompt=prompt,
                    changes=changes,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def filtering(criteria: str,
              include: bool = False,
              default: Optional[str] = None,
              prompt: str = "Filter the information from the text based on the filter criteria. Leave sentences unchanged if they are unrelated to the filter criteria:\n",
              examples: prm.Prompt = prm.Filter(),
              constraints: List[Callable] = [],
              limit: int = 1,
              stop: str | None = None,
              pre_processors: Optional[List[pre.PreProcessor]] = [pre.FilterPreProcessor()],
              post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
              **decorator_kwargs):
    """Filter information from a text based on a set of criteria."""
    return few_shot(prompt=prompt,
                    criteria=criteria,
                    include=include,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def map(instruction: str,
       default: str | None = None,
       prompt: str = "Transform each element in the input based on the instruction. Preserve container type and elements that don't match the instruction:\n",
       examples: prm.Prompt = prm.MapExpression(),
       constraints: list[Callable] = [],
       limit: int | None = 1,
       stop: str | None = None,
       pre_processors: list[pre.PreProcessor] | None = [pre.MapExpressionPreProcessor()],
       post_processors: list[post.PostProcessor] | None = [post.StripPostProcessor(), post.ASTPostProcessor()],
       **decorator_kwargs):
    """Semantic mapping operation that applies an instruction to each element in an iterable."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    context=instruction,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def notify(subscriber: Dict[str, Callable],
           default: Optional[object] = None,
           prompt: str = "List the semantically related topics:\n",
           examples: prm.Prompt = prm.SemanticMapping(),
           constraints: List[Callable] = [],
           limit: int | None = 1,
           stop: str | None = None,
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.SemanticMappingPreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.SplitPipePostProcessor(), post.NotifySubscriberPostProcessor()],
           **decorator_kwargs):
    """Notify subscribers based on a set of topics if detected in the input text and matching the key of the subscriber."""
    return few_shot(prompt=prompt,
                    subscriber=subscriber,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def compare(default: bool = False,
            operator: str = '>',
            prompt: str = "Compare 'A' and 'B' based on the operator:\n",
            examples: prm.Prompt = prm.CompareValues(),
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ComparePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Compare two objects based on the specified operator."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    max_tokens=None,
                    operator=operator,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def convert(format: str,
            default: Optional[str] = None,
            prompt: str = "Translate the following text into {} format.\n",
            examples: prm.Prompt = prm.Format(),
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.TextFormatPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Transformation operation from one format to another."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    format=format,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def transcribe(modify: str,
               default: Optional[str] = None,
               prompt: str = "Transcribe the following text by only modifying the text by the provided instruction.\n",
               examples: prm.Prompt = prm.Transcription(),
               constraints: List[Callable] = [],
               limit: int | None = 1,
               stop: str | None = None,
               pre_processors: Optional[List[pre.PreProcessor]] = [pre.TranscriptionPreProcessor()],
               post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
               **decorator_kwargs):
    """Transcription operation of a text to another styled text."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    modify=modify,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def style(description: str,
          libraries: List[str] = [],
          default: Optional[str] = None,
          prompt: str = "Style the [DATA] based on best practices and the descriptions in [...] brackets. Do not remove content from the data! Do not add libraries or other descriptions. \n",
          constraints: List[Callable] = [],
          limit: int | None = 1,
          stop: str | None = None,
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.StylePreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
          **decorator_kwargs):
    """Styles a given text based on best practices and a given description."""
    return few_shot(prompt=prompt,
                    libraries=libraries,
                    examples=None,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    description=description,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def analyze(query: str,
            exception: Exception,
            default: Optional[str] = None,
            prompt: str = "Only analyze the error message and suggest a potential correction, however, do NOT provide the code!\n",
            examples: prm.Prompt = prm.ExceptionMapping(),
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ExceptionPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Analyses an Exception and proposes a correction."""
    return few_shot(prompt=prompt,
                    query=query,
                    exception=exception,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def correct(context: str,
            exception: Exception,
            default: Optional[str] = None,
            prompt: str = "Correct the code according to the context description. Use markdown syntax to format the code; do not provide any other text.\n",
            examples: Optional[prm.Prompt] = None,
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.CorrectionPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.CodeExtractPostProcessor()],
            **decorator_kwargs):
    """Analyses an Exception and proposes a correction."""
    return few_shot(prompt=prompt,
                    context=context,
                    exception=exception,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def translate(language: str = 'English',
              default: str = "Sorry, I do not understand the given language.",
              prompt: str = "Your task is to translate and **only** translate the text into {}:\n",
              examples: Optional[prm.Prompt] = None,
              constraints: List[Callable] = [],
              limit: int | None = 1,
              stop: str | None = None,
              pre_processors: Optional[List[pre.PreProcessor]] = [pre.LanguagePreProcessor()],
              post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
              **decorator_kwargs):
    """Translates a given text into a specified language."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    language=language,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def rank(default: Optional[object] = None,
         order: str = 'desc',
         prompt: str = "Order the list of objects based on their quality measure and oder literal:\n",
         examples: prm.Prompt = prm.RankList(),
         constraints: List[Callable] = [],
         limit: int | None = 1,
         stop: str | None = None,
         pre_processors: Optional[List[pre.PreProcessor]] = [pre.RankPreProcessor()],
         post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.ASTPostProcessor()],
         **decorator_kwargs):
    """Ranks a list of objects based on their quality measure and order literal."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    order=order,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def replace(prompt: str = "Replace text parts by string pattern.\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.ReplaceText(),
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ReplacePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Replaces text parts by a given string pattern."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def include(prompt: str = "Include information based on description.\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.IncludeText(),
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.IncludePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Include information from a description."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def combine(prompt: str = "Add the two data types in a logical way:\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.CombineText(),
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.CombinePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Combines two data types in a logical way."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def negate(prompt: str = "Negate the following statement:\n",
           default: Optional[str] = None,
           examples: prm.Prompt = prm.NegateStatement(),
           constraints: List[Callable] = [],
           limit: int | None = 1,
           stop: str | None = None,
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.NegatePreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """Negates a given statement."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def contains(default: bool = False,
             prompt: str = "Is semantically the information of 'A' contained in 'B'?\n",
             examples: prm.Prompt = prm.ContainsValue(),
             constraints: List[Callable] = [],
             limit: int | None = 1,
             stop: str | None = None,
             pre_processors: Optional[List[pre.PreProcessor]] = [pre.ContainsPreProcessor()],
             post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
             **decorator_kwargs):
    """Determines whether a given string contains another string."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    max_tokens=None,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def isinstanceof(default: bool = False,
                 prompt: str = "Is 'A' an instance of 'B'?\n",
                 examples: prm.Prompt = prm.IsInstanceOf(),
                 constraints: List[Callable] = [],
                 limit: int | None = 1,
                 stop: str | None = None,
                 pre_processors: Optional[List[pre.PreProcessor]] = [pre.IsInstanceOfPreProcessor()],
                 post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
                 **decorator_kwargs):
    """Detects if one object is an instance of another."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    max_tokens=None,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def startswith(default: bool = False,
               prompt: str = "Does 'A' start with 'B'?\n",
               examples: prm.Prompt = prm.StartsWith(),
               constraints: List[Callable] = [],
               limit: int | None = 1,
               stop: str | None = None,
               pre_processors: Optional[List[pre.PreProcessor]] = [pre.StartsWithPreProcessor()],
               post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
               **decorator_kwargs):
    """Determines whether a string starts with a specified prefix."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    max_tokens=None,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def endswith(default: bool = False,
             prompt: str = "Does 'A' end with 'B'?\n",
             examples: prm.Prompt = prm.EndsWith(),
             constraints: List[Callable] = [],
             limit: int | None = 1,
             stop: str | None = None,
             pre_processors: Optional[List[pre.PreProcessor]] = [pre.EndsWithPreProcessor()],
             post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
             **decorator_kwargs):
    """Determines whether a string ends with a specified suffix."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    max_tokens=None,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def case(enum: List[str],
         default: str,
         prompt: str = "Classify the text according to one of the following categories and return only the category name: ",
         examples: Optional[prm.Prompt] = None,
         limit: int | None = 1,
         stop: str | None = None,
         pre_processors: Optional[List[pre.PreProcessor]] = [pre.EnumPreProcessor(), pre.TextMessagePreProcessor(), pre.PredictionMessagePreProcessor()],
         post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.CaseInsensitivePostProcessor()],
         **decorator_kwargs):
    """Classifies a text according to one of the given categories."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    enum=enum,
                    **decorator_kwargs)


def extract(prompt: str = "Extract a pattern from text:\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.ExtractPattern(),
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ExtractPatternPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Extracts a pattern from text."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)

def expression(prompt: str = "Evaluate the symbolic expressions:\n",
               default: Optional[str] = None,
               pre_processors: Optional[List[pre.PreProcessor]] = [],
               post_processors: Optional[List[post.PostProcessor]] = [post.WolframAlphaPostProcessor()],
               **decorator_kwargs):
    """Evaluates the symbolic expressions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['prompt'] = prompt
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='symbolic',
                                instance=instance,
                                func=func,
                                default=default,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def interpret(prompt: str = "Evaluate the symbolic expressions and return only the result:\n",
              default: Optional[str] = None,
              examples: prm.Prompt = prm.SimpleSymbolicExpression(),
              constraints: List[Callable] = [],
              limit: int | None = 1,
              stop: str | None = None,
              pre_processors: Optional[List[pre.PreProcessor]] = [pre.InterpretExpressionPreProcessor()],
              post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
              **decorator_kwargs):
    """Evaluates the symbolic expressions by interpreting the semantic meaning."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def logic(prompt: str = "Evaluate the logic expressions:\n",
          operator: str = 'and',
          default: Optional[str] = None,
          examples: prm.Prompt = prm.LogicExpression(),
          constraints: List[Callable] = [],
          limit: int | None = 1,
          stop: str | None = None,
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.LogicExpressionPreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
          **decorator_kwargs):
    """Evaluates a logic expression."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    operator=operator,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def invert(prompt: str = "Invert the logic of the content:\n",
           default: Optional[str] = None,
           examples: prm.Prompt = prm.InvertExpression(),
           constraints: List[Callable] = [],
           limit: int | None = 1,
           stop: str | None = None,
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.ArrowMessagePreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """Inverts the logic of a statement."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def simulate(prompt: str = "Simulate the following code:\n",
             default: Optional[str] = None,
             examples: prm.Prompt = prm.SimulateCode(),
             constraints: List[Callable] = [],
             limit: int | None = 1,
             stop: str | None = None,
             pre_processors: Optional[List[pre.PreProcessor]] = [pre.SimulateCodePreProcessor()],
             post_processors: Optional[List[post.PostProcessor]] = [post.SplitPipePostProcessor(), post.TakeLastPostProcessor()],
             **decorator_kwargs):
    """Simulates code and returns the result."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def code(prompt: str = "Generate code that solves the following problems:\n",
         default: Optional[str] = None,
         examples: prm.Prompt = prm.GenerateCode(),
         constraints: List[Callable] = [],
         limit: int | None = 1,
         stop: str | None = None,
         pre_processors: Optional[List[pre.PreProcessor]] = [pre.GenerateCodePreProcessor()],
         post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
         **decorator_kwargs):
    """Generates code that solves a given problem."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def outline(prompt: str = "Outline only the essential content as a short list of bullets. Each bullet is in a new line:\n",
            default: List[str] = None,
            examples: prm.Prompt = prm.TextToOutline(),
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.TextToOutlinePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.SplitNewLinePostProcessor()],
            **decorator_kwargs):
    """Outlines the essential content as a short list of bullets."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def unique(prompt: str = "Create a short unique key that captures the essential topic from the following statements and does not collide with the list of keys:\n",
           keys: List[str] = None,
           default: List[str] = None,
           examples: prm.Prompt = prm.UniqueKey(),
           constraints: List[Callable] = [],
           limit: int | None = 1,
           stop: str | None = None,
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.UniquePreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """Creates a short, unique key that captures the essential topic from the given statements and does not collide with the list of keys."""
    return few_shot(prompt=prompt,
                    keys=keys,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def clean(prompt: str = "Clean up the text from special characters or escape sequences. DO NOT change any words or sentences! Keep original semantics:\n",
          default: List[str] = None,
          examples: prm.Prompt = prm.CleanText(),
          constraints: List[Callable] = [],
          limit: int | None = 1,
          stop: str | None = None,
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.CleanTextMessagePreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
          **decorator_kwargs):
    """Cleans up a text from special characters and escape sequences."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def compose(prompt: str = "Create a coherent text based on the facts listed in the outline:\n",
            default: Optional[str] = None,
            examples: Optional[prm.Prompt] = None,
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.GenerateTextPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Compose a coherent text based on an outline."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def foreach(condition: str,
            apply: str,
            prompt: str = "Iterate over each element and apply operation based on condition:\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.ForEach(),
            constraints: List[Callable] = [],
            limit: int | None = 1,
            stop: str | None = None,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ForEachPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Applies an operation based on a given condition to each element in a list."""
    return few_shot(prompt=prompt,
                    condition=condition,
                    apply=apply,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def dictionary(context: str,
               prompt: str = "Map related content together under a common abstract topic. Do not remove content:\n",
               default: Optional[str] = None,
               examples: prm.Prompt = prm.MapContent(),
               constraints: List[Callable] = [],
               limit: int | None = 1,
               stop: str | None = None,
               pre_processors: Optional[List[pre.PreProcessor]] = [pre.MapPreProcessor()],
               post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.ASTPostProcessor()],
               **decorator_kwargs):
    """Maps related content together under a common abstract topic."""
    return few_shot(prompt=prompt,
                    context=context,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)

def listing(condition: str,
            prompt: str = "List each element contained in the text or list based on condition:\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.ListObjects(),
            constraints: List[Callable] = [],
            stop: str | None = None,
            limit: int | None = 1,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ListPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Lists each element contained in the text or list based on the given condition."""
    return few_shot(prompt=prompt,
                    condition=condition,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    stop=stop,
                    limit=limit,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def query(context: str,
          prompt: Optional[str] = None,
          examples: Optional[prm.Prompt] = None,
          constraints: List[Callable] = [],
          default: Optional[object] = None,
          stop: str | None = None,
          limit: int | None = 1,
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.QueryPreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
          **decorator_kwargs):
    """Performs a query given a context."""
    return few_shot(prompt=prompt,
                    context=context,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    stop=stop,
                    limit=limit,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def expand(prompt: Optional[str] = 'Write a self-contained function (with all imports) to solve a specific user problem task. Label the function with a name that describes the task.',
           examples: Optional[prm.Prompt] = prm.ExpandFunction(),
           constraints: List[Callable] = [],
           default: Optional[object] = None,
           stop: str | None = None,
           limit: int | None = 1,
           pre_processors: Optional[List[pre.PreProcessor]] = pre.ExpandFunctionPreProcessor(),
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.ExpandFunctionPostProcessor()],
           **decorator_kwargs):
    """Performs a expand command given a context to generate new prompts."""
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    stop=stop,
                    limit=limit,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def search(query: str,
           constraints: List[Callable] = [],
           default: Optional[object] = None,
           pre_processors: Optional[List[pre.PreProcessor]] = None,
           post_processors: Optional[List[post.PostProcessor]] = None,
           **decorator_kwargs):
    """Searches for a given query on the internet."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['query'] = query
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='search',
                                instance=instance,
                                func=func,
                                constraints=constraints,
                                default=default,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def opening(path: str,
            constraints: List[Callable] = [],
            default: Optional[object] = None,
            pre_processors: Optional[List[pre.PreProcessor]] = None,
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Opens a file and applies a given function to it."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['path'] = path
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='files',
                                instance=instance,
                                func=func,
                                constraints=constraints,
                                default=default,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def embed(entries: List[str],
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.UnwrapListSymbolsPreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = None,
          **decorator_kwargs):
    """Embeds the entries provided in a decorated function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['entries'] = entries
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='embedding',
                                instance=instance,
                                func=func,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def cluster(entries: List[str],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.UnwrapListSymbolsPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.ClusterPostProcessor()],
            **decorator_kwargs):
    """Embeds and clusters the input entries."""
    assert any(isinstance(pr, post.ClusterPostProcessor) for pr in post_processors), "At least one post processor must be a 'ClusterPostProcessor' for clustering!"
    for post_pr in post_processors:
        if isinstance(post_pr, post.ClusterPostProcessor):
            post_pr.set(decorator_kwargs)

    return embed(entries=entries,
                 pre_processors=pre_processors,
                 post_processors=post_processors,
                 **decorator_kwargs)


def draw(operation: str = 'create',
         prompt: str = '',
         pre_processors: Optional[List[pre.PreProcessor]] = [pre.ValuePreProcessor()],
         post_processors: Optional[List[post.PostProcessor]] = None,
         **decorator_kwargs):
    """Draws an image provided in a decorated function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['operation'] = operation
            decorator_kwargs['prompt']    = prompt
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='drawing',
                                instance=instance,
                                func=func,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def text_vision(image: Optional[str|bytes] = None,
                text: List[str] = None,
                pre_processors: Optional[List[pre.PreProcessor]] = None,
                post_processors: Optional[List[post.PostProcessor]] = None,
                **decorator_kwargs):
    """Performs vision-related associative tasks. Currently limited to CLIP model embeddings."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['image'] = image
            decorator_kwargs['text']  = text
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='text_vision',
                                instance=instance,
                                func=func,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def ocr(image: str,
        pre_processors: Optional[List[pre.PreProcessor]] = None,
        post_processors: Optional[List[post.PostProcessor]] = None,
        **decorator_kwargs):
    """Performs Optical Character Recognition (OCR) on an image."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['image'] = image
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='ocr',
                                instance=instance,
                                func=func,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def speech_to_text(prompt: str = 'decode',
                   pre_processors: Optional[List[pre.PreProcessor]] = None,
                   post_processors: Optional[List[post.PostProcessor]] = None,
                   **decorator_kwargs):
    """Decorates the given function for speech recognition."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['prompt'] = prompt
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='speech-to-text',
                                instance=instance,
                                func=func,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def text_to_speech(prompt: str,
                   path: str,
                   voice: str = 'nova',
                   pre_processors: Optional[List[pre.PreProcessor]] = None,
                   post_processors: Optional[List[post.PostProcessor]] = None,
                   **decorator_kwargs):
    """Decorates the given function for text to speech synthesis."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['path']   = path
            decorator_kwargs['voice']  = voice
            decorator_kwargs['prompt'] = prompt
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='text-to-speech',
                                instance=instance,
                                func=func,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def output(constraints: List[Callable] = [],
           default: Optional[object] = None,
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.ConsolePreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.ConsolePostProcessor()],
           **decorator_kwargs):
    """Offers an output stream for writing results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='output',
                                instance=instance,
                                func=func,
                                constraints=constraints,
                                default=default,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def fetch(url: str,
          pattern: str = ' ',
          constraints: List[Callable] = [],
          default: Optional[object] = None,
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.CrawlPatternPreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [],
          **decorator_kwargs):
    """Fetches data from a given URL and applies the provided post-processors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['urls'] = url
            decorator_kwargs['patterns'] = pattern
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='crawler',
                                instance=instance,
                                func=func,
                                constraints=constraints,
                                default=default,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def userinput(constraints: List[Callable] = [],
              default: Optional[object] = None,
              pre_processors: Optional[List[pre.PreProcessor]] = [pre.ConsoleInputPreProcessor()],
              post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
              **decorator_kwargs):
    """Prompts for user input and returns the user response through a decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='userinput',
                                instance=instance,
                                func=func,
                                constraints=constraints,
                                default=default,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def execute(default: Optional[str] = None,
            constraints: List[Callable] = [],
            pre_processors: List[pre.PreProcessor] = [],
            post_processors: List[post.PostProcessor] = [],
            **decorator_kwargs):
    """Executes a given function after applying constraints, pre-processing and post-processing."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='execute',
                                instance=instance,
                                func=func,
                                constraints=constraints,
                                default=default,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def index(prompt: Any,
          index_name: str,
          operation: str = 'search', # | add | config
          default: Optional[str] = None,
          constraints: List[Callable] = [],
          pre_processors: List[pre.PreProcessor] = [],
          post_processors: List[post.PostProcessor] = [],
          **decorator_kwargs):
    """Query for a given index and returns the result through a decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['operation']  = operation
            decorator_kwargs['prompt']     = prompt
            decorator_kwargs['index_name'] = index_name
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='index',
                                instance=instance,
                                func=func,
                                constraints=constraints,
                                default=default,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def command(engines: List[str] = ['all'], **decorator_kwargs):
    """Decorates a function to forward commands to the engine backends."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance):
            return EngineRepository.command(
                    engines=engines,
                    instance=instance,
                    func=func,
                    **decorator_kwargs
                )
        return wrapper
    return decorator


def register(engines: Dict[str, Any]):
    """Decorates a function to initialize custom engines as backends."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, **kwargs):
            return EngineRepository.register(
                    engines=engines,
                    engine_instance=instance,
                    func=func,
                    **kwargs
                )
        return wrapper
    return decorator


def tune(operation: str = 'create',
         pre_processors: Optional[List[pre.PreProcessor]] = None,
         post_processors: Optional[List[post.PostProcessor]] = None,
         **decorator_kwargs):
    """Fine tune a LLM."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            decorator_kwargs['__cmd__'] = operation #TODO: update engine
            # Construct container object for the arguments and kwargs
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='finetune',
                                instance=instance,
                                func=func,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def caption(image: str,
            prompt: str,
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ValuePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = None,
            **decorator_kwargs):
    """Caption the content of an image."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['image']  = image
            decorator_kwargs['prompt'] = prompt
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='imagecaptioning',
                                instance=instance,
                                func=func,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator
