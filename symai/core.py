import functools
from collections.abc import Callable
from typing import Any

from box import Box

from . import post_processors as post
from . import pre_processors as pre
from . import prompts as prm
from .functional import EngineRepository
from .symbol import Expression, Metadata

# Module-level singletons used to provide default prompt and processor instances.
_PREPROCESSOR_EXPAND_FUNCTION = pre.ExpandFunctionPreProcessor()
_PROMPT_CLEAN_TEXT = prm.CleanText()
_PROMPT_COMBINE_TEXT = prm.CombineText()
_PROMPT_COMPARE_VALUES = prm.CompareValues()
_PROMPT_CONTAINS_VALUE = prm.ContainsValue()
_PROMPT_ENDS_WITH = prm.EndsWith()
_PROMPT_EXCEPTION_MAPPING = prm.ExceptionMapping()
_PROMPT_EXPAND_FUNCTION = prm.ExpandFunction()
_PROMPT_EXTRACT_PATTERN = prm.ExtractPattern()
_PROMPT_FILTER = prm.Filter()
_PROMPT_FOR_EACH = prm.ForEach()
_PROMPT_FORMAT = prm.Format()
_PROMPT_FUZZY_EQUALS = prm.FuzzyEquals()
_PROMPT_GENERATE_CODE = prm.GenerateCode()
_PROMPT_INCLUDE_TEXT = prm.IncludeText()
_PROMPT_INDEX = prm.Index()
_PROMPT_INVERT_EXPRESSION = prm.InvertExpression()
_PROMPT_IS_INSTANCE_OF = prm.IsInstanceOf()
_PROMPT_LIST_OBJECTS = prm.ListObjects()
_PROMPT_LOGIC_EXPRESSION = prm.LogicExpression()
_PROMPT_MAP_CONTENT = prm.MapContent()
_PROMPT_MAP_EXPRESSION = prm.MapExpression()
_PROMPT_MODIFY = prm.Modify()
_PROMPT_NEGATE_STATEMENT = prm.NegateStatement()
_PROMPT_RANK_LIST = prm.RankList()
_PROMPT_REMOVE_INDEX = prm.RemoveIndex()
_PROMPT_REPLACE_TEXT = prm.ReplaceText()
_PROMPT_SEMANTIC_MAPPING = prm.SemanticMapping()
_PROMPT_SET_INDEX = prm.SetIndex()
_PROMPT_SIMPLE_SYMBOLIC_EXPRESSION = prm.SimpleSymbolicExpression()
_PROMPT_SIMULATE_CODE = prm.SimulateCode()
_PROMPT_STARTS_WITH = prm.StartsWith()
_PROMPT_SUFFICIENT_INFORMATION = prm.SufficientInformation()
_PROMPT_TEXT_TO_OUTLINE = prm.TextToOutline()
_PROMPT_TRANSCRIPTION = prm.Transcription()
_PROMPT_UNIQUE_KEY = prm.UniqueKey()


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
        self._apply_default_properties()

    def _set_all_kwargs_as_properties(self):
        for key, value in self.kwargs.items():
            setattr(self.prop, key, value)

    def _apply_default_properties(self):
        # Set default values if not specified for backend processing
        # Reserved keywords
        default_properties = {
            # used for previewing the input (also for operators)
            'preview': Argument._default_preview_value,
            'raw_input': False,
            'raw_output': False,
            'return_metadata': False,
            'logging': False,
            'verbose': False,
            'self_prompt': False,
            'truncation_percentage': None,
            'truncation_type': 'tail',
            'response_format': None,
            'log_level': None,
            'time_clock': None,
            'payload': None,
            'processed_input': None,
            'template_suffix': None,
            'input_handler': None,
            'output_handler': None,
            'suppress_verbose_output': Argument._default_suppress_verbose_output,
            'parse_system_instructions': Argument._default_parse_system_instructions,
        }
        for key, default_value in default_properties.items():
            if key not in self.kwargs:
                setattr(self.prop, key, default_value)

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
             examples: prm.Prompt = None,
             constraints: list[Callable] | None = None,
             default: Any = None,
             limit: int = 1,
             pre_processors: list[pre.PreProcessor] | None = None,
             post_processors: list[post.PostProcessor] | None = None,
             **decorator_kwargs):
    """"General decorator for the neural processing engine."""
    if constraints is None:
        constraints = []
    if examples is None:
        examples = []
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
              constraints: list[Callable] | None = None,
              default: object | None = None,
              limit: int = 1,
              pre_processors: list[pre.PreProcessor] | None = None,
              post_processors: list[post.PostProcessor] | None = None,
              **decorator_kwargs):
    """"General decorator for the neural processing engine."""
    if constraints is None:
        constraints = []
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
              context: str | None = None,
              constraints: list[Callable] | None = None,
              default: object | None = None,
              limit: int = 1,
              stop: str | list[str] = '',
              pre_processors: list[pre.PreProcessor] | None = None,
              post_processors: list[post.PostProcessor] | None = None,
              **decorator_kwargs):
    """Summarizes the content of a text."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.SummaryPreProcessing()]
    if constraints is None:
        constraints = []
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
           examples: prm.Prompt = _PROMPT_FUZZY_EQUALS,
           constraints: list[Callable] | None = None,
           limit: int = 1,
           stop: str | list[str] = '',
           pre_processors: list[pre.PreProcessor] | None = None,
           post_processors: list[post.PostProcessor] | None = None,
           **decorator_kwargs):
    """Equality function for two objects."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.EqualsPreProcessor()]
    if constraints is None:
        constraints = []
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
               examples: prm.Prompt = _PROMPT_SUFFICIENT_INFORMATION,
               constraints: list[Callable] | None = None,
               limit: int = 1,
               stop: str | list[str] = '',
               pre_processors: list[pre.PreProcessor] | None = None,
               post_processors: list[post.PostProcessor] | None = None,
               **decorator_kwargs) -> bool:
    """Determines if there is sufficient information to answer the given query."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.SufficientInformationPreProcessor()]
    if constraints is None:
        constraints = []
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


def delitem(default: str | None = None,
            prompt: str = "Delete the items at the index position:\n",
            examples: prm.Prompt = _PROMPT_REMOVE_INDEX,
            constraints: list[Callable] | None = None,
            limit: int = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Deletes the items at the specified index position."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.DeleteIndexPreProcessor()]
    if constraints is None:
        constraints = []
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def setitem(default: str | None = None,
            prompt: str = "Set item at index position:\n",
            examples: prm.Prompt = _PROMPT_SET_INDEX,
            constraints: list[Callable] | None = None,
            limit: int = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Sets an item at a given index position in a sequence."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.SetIndexPreProcessor()]
    if constraints is None:
        constraints = []
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def getitem(default: str | None = None,
            prompt: str = "Get item at index position:\n",
            examples: prm.Prompt = _PROMPT_INDEX,
            constraints: list[Callable] | None = None,
            limit: int = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Retrieves the item at the given index position."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.IndexPreProcessor()]
    if constraints is None:
        constraints = []
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
           default: str | None = None,
           prompt: str = "Modify the text to match the criteria:\n",
           examples: prm.Prompt = _PROMPT_MODIFY,
           constraints: list[Callable] | None = None,
           limit: int = 1,
           stop: str | list[str] = '',
           pre_processors: list[pre.PreProcessor] | None = None,
           post_processors: list[post.PostProcessor] | None = None,
           **decorator_kwargs):
    """A function to modify a text based on a set of criteria."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ModifyPreProcessor()]
    if constraints is None:
        constraints = []
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
              default: str | None = None,
              prompt: str = "Filter the information from the text based on the filter criteria. Leave sentences unchanged if they are unrelated to the filter criteria:\n",
              examples: prm.Prompt = _PROMPT_FILTER,
              constraints: list[Callable] | None = None,
              limit: int = 1,
              stop: str | list[str] = '',
              pre_processors: list[pre.PreProcessor] | None = None,
              post_processors: list[post.PostProcessor] | None = None,
              **decorator_kwargs):
    """Filter information from a text based on a set of criteria."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.FilterPreProcessor()]
    if constraints is None:
        constraints = []
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


def map( # noqa: A001
        instruction: str,
        default: str | None = None,
        prompt: str = "Transform each element in the input based on the instruction. Preserve container type and elements that don't match the instruction:\n",
        examples: prm.Prompt = _PROMPT_MAP_EXPRESSION,
        constraints: list[Callable] | None = None,
        limit: int | None = 1,
        stop: str | list[str] = '',
        pre_processors: list[pre.PreProcessor] | None = None,
        post_processors: list[post.PostProcessor] | None = None,
        **decorator_kwargs):
    """Semantic mapping operation that applies an instruction to each element in an iterable."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor(), post.ASTPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.MapExpressionPreProcessor()]
    if constraints is None:
        constraints = []
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


def notify(subscriber: dict[str, Callable],
           default: object | None = None,
           prompt: str = "List the semantically related topics:\n",
           examples: prm.Prompt = _PROMPT_SEMANTIC_MAPPING,
           constraints: list[Callable] | None = None,
           limit: int | None = 1,
           stop: str | list[str] = '',
           pre_processors: list[pre.PreProcessor] | None = None,
           post_processors: list[post.PostProcessor] | None = None,
           **decorator_kwargs):
    """Notify subscribers based on a set of topics if detected in the input text and matching the key of the subscriber."""
    if post_processors is None:
        post_processors = [post.SplitPipePostProcessor(), post.NotifySubscriberPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.SemanticMappingPreProcessor()]
    if constraints is None:
        constraints = []
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
            examples: prm.Prompt = _PROMPT_COMPARE_VALUES,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Compare two objects based on the specified operator."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ComparePreProcessor()]
    if constraints is None:
        constraints = []
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


def convert(
        format: str,
        default: str | None = None,
        prompt: str = "Translate the following text into {} format.\n",
        examples: prm.Prompt = _PROMPT_FORMAT,
        constraints: list[Callable] | None = None,
        limit: int | None = 1,
        stop: str | list[str] = '',
        pre_processors: list[pre.PreProcessor] | None = None,
        post_processors: list[post.PostProcessor] | None = None,
        **decorator_kwargs):
    """Transformation operation from one format to another."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.TextFormatPreProcessor()]
    if constraints is None:
        constraints = []
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
               default: str | None = None,
               prompt: str = "Transcribe the following text by only modifying the text by the provided instruction.\n",
               examples: prm.Prompt = _PROMPT_TRANSCRIPTION,
               constraints: list[Callable] | None = None,
               limit: int | None = 1,
               stop: str | list[str] = '',
               pre_processors: list[pre.PreProcessor] | None = None,
               post_processors: list[post.PostProcessor] | None = None,
               **decorator_kwargs):
    """Transcription operation of a text to another styled text."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.TranscriptionPreProcessor()]
    if constraints is None:
        constraints = []
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
          libraries: list[str] | None = None,
          default: str | None = None,
          prompt: str = "Style the [DATA] based on best practices and the descriptions in [...] brackets. Do not remove content from the data! Do not add libraries or other descriptions. \n",
          constraints: list[Callable] | None = None,
          limit: int | None = 1,
          stop: str | list[str] = '',
          pre_processors: list[pre.PreProcessor] | None = None,
          post_processors: list[post.PostProcessor] | None = None,
          **decorator_kwargs):
    """Styles a given text based on best practices and a given description."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.StylePreProcessor()]
    if constraints is None:
        constraints = []
    if libraries is None:
        libraries = []
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
            default: str | None = None,
            prompt: str = "Only analyze the error message and suggest a potential correction, however, do NOT provide the code!\n",
            examples: prm.Prompt = _PROMPT_EXCEPTION_MAPPING,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Analyses an Exception and proposes a correction."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ExceptionPreProcessor()]
    if constraints is None:
        constraints = []
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
            default: str | None = None,
            prompt: str = "Correct the code according to the context description. Use markdown syntax to format the code; do not provide any other text.\n",
            examples: prm.Prompt | None = None,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Analyses an Exception and proposes a correction."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor(), post.CodeExtractPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.CorrectionPreProcessor()]
    if constraints is None:
        constraints = []
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
              examples: prm.Prompt | None = None,
              constraints: list[Callable] | None = None,
              limit: int | None = 1,
              stop: str | list[str] = '',
              pre_processors: list[pre.PreProcessor] | None = None,
              post_processors: list[post.PostProcessor] | None = None,
              **decorator_kwargs):
    """Translates a given text into a specified language."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.LanguagePreProcessor()]
    if constraints is None:
        constraints = []
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


def rank(default: object | None = None,
         order: str = 'desc',
         prompt: str = "Order the list of objects based on their quality measure and oder literal:\n",
         examples: prm.Prompt = _PROMPT_RANK_LIST,
         constraints: list[Callable] | None = None,
         limit: int | None = 1,
         stop: str | list[str] = '',
         pre_processors: list[pre.PreProcessor] | None = None,
         post_processors: list[post.PostProcessor] | None = None,
         **decorator_kwargs):
    """Ranks a list of objects based on their quality measure and order literal."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor(), post.ASTPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.RankPreProcessor()]
    if constraints is None:
        constraints = []
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
            default: str | None = None,
            examples: prm.Prompt = _PROMPT_REPLACE_TEXT,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Replaces text parts by a given string pattern."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ReplacePreProcessor()]
    if constraints is None:
        constraints = []
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
            default: str | None = None,
            examples: prm.Prompt = _PROMPT_INCLUDE_TEXT,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Include information from a description."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.IncludePreProcessor()]
    if constraints is None:
        constraints = []
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
            default: str | None = None,
            examples: prm.Prompt = _PROMPT_COMBINE_TEXT,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Combines two data types in a logical way."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.CombinePreProcessor()]
    if constraints is None:
        constraints = []
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
           default: str | None = None,
           examples: prm.Prompt = _PROMPT_NEGATE_STATEMENT,
           constraints: list[Callable] | None = None,
           limit: int | None = 1,
           stop: str | list[str] = '',
           pre_processors: list[pre.PreProcessor] | None = None,
           post_processors: list[post.PostProcessor] | None = None,
           **decorator_kwargs):
    """Negates a given statement."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.NegatePreProcessor()]
    if constraints is None:
        constraints = []
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
             examples: prm.Prompt = _PROMPT_CONTAINS_VALUE,
             constraints: list[Callable] | None = None,
             limit: int | None = 1,
             stop: str | list[str] = '',
             pre_processors: list[pre.PreProcessor] | None = None,
             post_processors: list[post.PostProcessor] | None = None,
             **decorator_kwargs):
    """Determines whether a given string contains another string."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ContainsPreProcessor()]
    if constraints is None:
        constraints = []
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
                 examples: prm.Prompt = _PROMPT_IS_INSTANCE_OF,
                 constraints: list[Callable] | None = None,
                 limit: int | None = 1,
                 stop: str | list[str] = '',
                 pre_processors: list[pre.PreProcessor] | None = None,
                 post_processors: list[post.PostProcessor] | None = None,
                 **decorator_kwargs):
    """Detects if one object is an instance of another."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.IsInstanceOfPreProcessor()]
    if constraints is None:
        constraints = []
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
               examples: prm.Prompt = _PROMPT_STARTS_WITH,
               constraints: list[Callable] | None = None,
               limit: int | None = 1,
               stop: str | list[str] = '',
               pre_processors: list[pre.PreProcessor] | None = None,
               post_processors: list[post.PostProcessor] | None = None,
               **decorator_kwargs):
    """Determines whether a string starts with a specified prefix."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.StartsWithPreProcessor()]
    if constraints is None:
        constraints = []
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
             examples: prm.Prompt = _PROMPT_ENDS_WITH,
             constraints: list[Callable] | None = None,
             limit: int | None = 1,
             stop: str | list[str] = '',
             pre_processors: list[pre.PreProcessor] | None = None,
             post_processors: list[post.PostProcessor] | None = None,
             **decorator_kwargs):
    """Determines whether a string ends with a specified suffix."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.EndsWithPreProcessor()]
    if constraints is None:
        constraints = []
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


def case(enum: list[str],
         default: str,
         prompt: str = "Classify the text according to one of the following categories and return only the category name: ",
         examples: prm.Prompt | None = None,
         limit: int | None = 1,
         stop: str | list[str] = '',
         pre_processors: list[pre.PreProcessor] | None = None,
         post_processors: list[post.PostProcessor] | None = None,
         **decorator_kwargs):
    """Classifies a text according to one of the given categories."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor(), post.CaseInsensitivePostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.EnumPreProcessor(), pre.TextMessagePreProcessor(), pre.PredictionMessagePreProcessor()]
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
            default: str | None = None,
            examples: prm.Prompt = _PROMPT_EXTRACT_PATTERN,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Extracts a pattern from text."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ExtractPatternPreProcessor()]
    if constraints is None:
        constraints = []
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
               default: str | None = None,
               pre_processors: list[pre.PreProcessor] | None = None,
               post_processors: list[post.PostProcessor] | None = None,
               **decorator_kwargs):
    """Evaluates the symbolic expressions."""
    if post_processors is None:
        post_processors = [post.WolframAlphaPostProcessor()]
    if pre_processors is None:
        pre_processors = []
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
              default: str | None = None,
              examples: prm.Prompt = _PROMPT_SIMPLE_SYMBOLIC_EXPRESSION,
              constraints: list[Callable] | None = None,
              limit: int | None = 1,
              stop: str | list[str] = '',
              pre_processors: list[pre.PreProcessor] | None = None,
              post_processors: list[post.PostProcessor] | None = None,
              **decorator_kwargs):
    """Evaluates the symbolic expressions by interpreting the semantic meaning."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.InterpretExpressionPreProcessor()]
    if constraints is None:
        constraints = []
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
          default: str | None = None,
          examples: prm.Prompt = _PROMPT_LOGIC_EXPRESSION,
          constraints: list[Callable] | None = None,
          limit: int | None = 1,
          stop: str | list[str] = '',
          pre_processors: list[pre.PreProcessor] | None = None,
          post_processors: list[post.PostProcessor] | None = None,
          **decorator_kwargs):
    """Evaluates a logic expression."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.LogicExpressionPreProcessor()]
    if constraints is None:
        constraints = []
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
           default: str | None = None,
           examples: prm.Prompt = _PROMPT_INVERT_EXPRESSION,
           constraints: list[Callable] | None = None,
           limit: int | None = 1,
           stop: str | list[str] = '',
           pre_processors: list[pre.PreProcessor] | None = None,
           post_processors: list[post.PostProcessor] | None = None,
           **decorator_kwargs):
    """Inverts the logic of a statement."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ArrowMessagePreProcessor()]
    if constraints is None:
        constraints = []
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
             default: str | None = None,
             examples: prm.Prompt = _PROMPT_SIMULATE_CODE,
             constraints: list[Callable] | None = None,
             limit: int | None = 1,
             stop: str | list[str] = '',
             pre_processors: list[pre.PreProcessor] | None = None,
             post_processors: list[post.PostProcessor] | None = None,
             **decorator_kwargs):
    """Simulates code and returns the result."""
    if post_processors is None:
        post_processors = [post.SplitPipePostProcessor(), post.TakeLastPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.SimulateCodePreProcessor()]
    if constraints is None:
        constraints = []
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
         default: str | None = None,
         examples: prm.Prompt = _PROMPT_GENERATE_CODE,
         constraints: list[Callable] | None = None,
         limit: int | None = 1,
         stop: str | list[str] = '',
         pre_processors: list[pre.PreProcessor] | None = None,
         post_processors: list[post.PostProcessor] | None = None,
         **decorator_kwargs):
    """Generates code that solves a given problem."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.GenerateCodePreProcessor()]
    if constraints is None:
        constraints = []
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
            default: list[str] | None = None,
            examples: prm.Prompt = _PROMPT_TEXT_TO_OUTLINE,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Outlines the essential content as a short list of bullets."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor(), post.SplitNewLinePostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.TextToOutlinePreProcessor()]
    if constraints is None:
        constraints = []
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
           keys: list[str] | None = None,
           default: list[str] | None = None,
           examples: prm.Prompt = _PROMPT_UNIQUE_KEY,
           constraints: list[Callable] | None = None,
           limit: int | None = 1,
           stop: str | list[str] = '',
           pre_processors: list[pre.PreProcessor] | None = None,
           post_processors: list[post.PostProcessor] | None = None,
           **decorator_kwargs):
    """Creates a short, unique key that captures the essential topic from the given statements and does not collide with the list of keys."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.UniquePreProcessor()]
    if constraints is None:
        constraints = []
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
          default: list[str] | None = None,
          examples: prm.Prompt = _PROMPT_CLEAN_TEXT,
          constraints: list[Callable] | None = None,
          limit: int | None = 1,
          stop: str | list[str] = '',
          pre_processors: list[pre.PreProcessor] | None = None,
          post_processors: list[post.PostProcessor] | None = None,
          **decorator_kwargs):
    """Cleans up a text from special characters and escape sequences."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.CleanTextMessagePreProcessor()]
    if constraints is None:
        constraints = []
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
            default: str | None = None,
            examples: prm.Prompt | None = None,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Compose a coherent text based on an outline."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.GenerateTextPreProcessor()]
    if constraints is None:
        constraints = []
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
            default: str | None = None,
            examples: prm.Prompt = _PROMPT_FOR_EACH,
            constraints: list[Callable] | None = None,
            limit: int | None = 1,
            stop: str | list[str] = '',
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Applies an operation based on a given condition to each element in a list."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ForEachPreProcessor()]
    if constraints is None:
        constraints = []
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
               default: str | None = None,
               examples: prm.Prompt = _PROMPT_MAP_CONTENT,
               constraints: list[Callable] | None = None,
               limit: int | None = 1,
               stop: str | list[str] = '',
               pre_processors: list[pre.PreProcessor] | None = None,
               post_processors: list[post.PostProcessor] | None = None,
               **decorator_kwargs):
    """Maps related content together under a common abstract topic."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor(), post.ASTPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.MapPreProcessor()]
    if constraints is None:
        constraints = []
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
            default: str | None = None,
            examples: prm.Prompt = _PROMPT_LIST_OBJECTS,
            constraints: list[Callable] | None = None,
            stop: str | list[str] = '',
            limit: int | None = 1,
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Lists each element contained in the text or list based on the given condition."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ListPreProcessor()]
    if constraints is None:
        constraints = []
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
          prompt: str | None = None,
          examples: prm.Prompt | None = None,
          constraints: list[Callable] | None = None,
          default: object | None = None,
          stop: str | list[str] = '',
          limit: int | None = 1,
          pre_processors: list[pre.PreProcessor] | None = None,
          post_processors: list[post.PostProcessor] | None = None,
          **decorator_kwargs):
    """Performs a query given a context."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.QueryPreProcessor()]
    if constraints is None:
        constraints = []
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


def expand(prompt: str | None = 'Write a self-contained function (with all imports) to solve a specific user problem task. Label the function with a name that describes the task.',
           examples: prm.Prompt | None = _PROMPT_EXPAND_FUNCTION,
           constraints: list[Callable] | None = None,
           default: object | None = None,
           stop: str | list[str] = '',
           limit: int | None = 1,
           pre_processors: list[pre.PreProcessor] | None = _PREPROCESSOR_EXPAND_FUNCTION,
           post_processors: list[post.PostProcessor] | None = None,
           **decorator_kwargs):
    """Performs a expand command given a context to generate new prompts."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor(), post.ExpandFunctionPostProcessor()]
    if constraints is None:
        constraints = []
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
           constraints: list[Callable] | None = None,
           default: object | None = None,
           pre_processors: list[pre.PreProcessor] | None = None,
           post_processors: list[post.PostProcessor] | None = None,
           **decorator_kwargs):
    """Searches for a given query on the internet."""
    if constraints is None:
        constraints = []
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
            constraints: list[Callable] | None = None,
            default: object | None = None,
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Opens a file and applies a given function to it."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if constraints is None:
        constraints = []
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


def embed(entries: list[str],
          pre_processors: list[pre.PreProcessor] | None = None,
          post_processors: list[post.PostProcessor] | None = None,
          **decorator_kwargs):
    """Embeds the entries provided in a decorated function."""
    if pre_processors is None:
        pre_processors = [pre.UnwrapListSymbolsPreProcessor()]
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


def cluster(entries: list[str],
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Embeds and clusters the input entries."""
    if post_processors is None:
        post_processors = [post.ClusterPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.UnwrapListSymbolsPreProcessor()]
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
         pre_processors: list[pre.PreProcessor] | None = None,
         post_processors: list[post.PostProcessor] | None = None,
         **decorator_kwargs):
    """Draws an image provided in a decorated function."""
    if pre_processors is None:
        pre_processors = [pre.ValuePreProcessor()]
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


def text_vision(image: str | bytes | None = None,
                text: list[str] | None = None,
                pre_processors: list[pre.PreProcessor] | None = None,
                post_processors: list[post.PostProcessor] | None = None,
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
        pre_processors: list[pre.PreProcessor] | None = None,
        post_processors: list[post.PostProcessor] | None = None,
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
                   pre_processors: list[pre.PreProcessor] | None = None,
                   post_processors: list[post.PostProcessor] | None = None,
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
                   pre_processors: list[pre.PreProcessor] | None = None,
                   post_processors: list[post.PostProcessor] | None = None,
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


def output(constraints: list[Callable] | None = None,
           default: object | None = None,
           pre_processors: list[pre.PreProcessor] | None = None,
           post_processors: list[post.PostProcessor] | None = None,
           **decorator_kwargs):
    """Offers an output stream for writing results."""
    if post_processors is None:
        post_processors = [post.ConsolePostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ConsolePreProcessor()]
    if constraints is None:
        constraints = []
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


def scrape(url: str,
          constraints: list[Callable] | None = None,
          default: object | None = None,
          pre_processors: list[pre.PreProcessor] | None = None,
          post_processors: list[post.PostProcessor] | None = None,
          **decorator_kwargs):
    """Fetches data from a given URL and applies the provided post-processors."""
    if post_processors is None:
        post_processors = []
    if pre_processors is None:
        pre_processors = []
    if constraints is None:
        constraints = []
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            decorator_kwargs['url'] = url
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='webscraping',
                                instance=instance,
                                func=func,
                                constraints=constraints,
                                default=default,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def userinput(constraints: list[Callable] | None = None,
              default: object | None = None,
              pre_processors: list[pre.PreProcessor] | None = None,
              post_processors: list[post.PostProcessor] | None = None,
              **decorator_kwargs):
    """Prompts for user input and returns the user response through a decorator."""
    if post_processors is None:
        post_processors = [post.StripPostProcessor()]
    if pre_processors is None:
        pre_processors = [pre.ConsoleInputPreProcessor()]
    if constraints is None:
        constraints = []
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


def execute(default: str | None = None,
            constraints: list[Callable] | None = None,
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Executes a given function after applying constraints, pre-processing and post-processing."""
    if post_processors is None:
        post_processors = []
    if pre_processors is None:
        pre_processors = []
    if constraints is None:
        constraints = []
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
          default: str | None = None,
          constraints: list[Callable] | None = None,
          pre_processors: list[pre.PreProcessor] | None = None,
          post_processors: list[post.PostProcessor] | None = None,
          **decorator_kwargs):
    """Query for a given index and returns the result through a decorator."""
    if post_processors is None:
        post_processors = []
    if pre_processors is None:
        pre_processors = []
    if constraints is None:
        constraints = []
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


def command(engines: list[str] | None = None, **decorator_kwargs):
    """Decorates a function to forward commands to the engine backends."""
    if engines is None:
        engines = ['all']
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


def register(engines: dict[str, Any]):
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
         pre_processors: list[pre.PreProcessor] | None = None,
         post_processors: list[post.PostProcessor] | None = None,
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
            pre_processors: list[pre.PreProcessor] | None = None,
            post_processors: list[post.PostProcessor] | None = None,
            **decorator_kwargs):
    """Caption the content of an image."""
    if pre_processors is None:
        pre_processors = [pre.ValuePreProcessor()]
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
