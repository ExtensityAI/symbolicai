import functools

from box import Box
from typing import Callable, Dict, List, Optional, Any

from . import post_processors as post
from . import pre_processors as pre
from . import prompts as prm
from .functional import EngineRepository
from .symbol import Expression, Metadata



class Argument(Expression):
    _default_suppress_verbose_output            = False
    _default_parse_system_instructions          = False

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
        if 'preview' not in self.kwargs:
            self.prop.preview           = False
        if 'raw_input' not in self.kwargs:
            self.prop.raw_input         = False
        if 'raw_output' not in self.kwargs:
            self.prop.raw_output        = False
        if 'logging' not in self.kwargs:
            self.prop.logging           = False
        if 'verbose' not in self.kwargs:
            self.prop.verbose           = False
        if 'response_format' not in self.kwargs:
            self.prop.response_format   = None
        if 'log_level' not in self.kwargs:
            self.prop.log_level         = None
        if 'time_clock' not in self.kwargs:
            self.prop.time_clock        = None
        if 'payload' not in self.kwargs:
            self.prop.payload           = None
        if 'processed_input' not in self.kwargs:
            self.prop.processed_input   = None
        if 'template_suffix' not in self.kwargs:
            self.prop.template_suffix   = None
        if 'input_handler' not in self.kwargs:
            self.prop.input_handler     = None
        if 'output_handler' not in self.kwargs:
            self.prop.output_handler    = None
        if 'suppress_verbose_output' not in self.kwargs:
            self.prop.suppress_verbose_output    = Argument._default_suppress_verbose_output
        if 'parse_system_instructions' not in self.kwargs:
            self.prop.parse_system_instructions  = Argument._default_parse_system_instructions

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
    """"General decorator for the neural processing engine.
    This method is used to decorate functions which can build any expression in a examples-based way.

    Args:
        prompt (str): The prompt describing the task. Defaults to 'Summarize the content of the following text:\n'.
        examples (Any): A object containing examples to be used for the task in specified format.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of results to be returned, if more are obtained.
        default (object, optional): Default value if prediction fails. Defaults to None.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model to match the format of the examples. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        **decorator_kwargs: Additional arguments as key-value pairs passed to the decorated function, which can later accessed in pre_processors and post_processors via the argument.kwargs['key'] dictionary.

    Returns:
        object: The prediction of the model based on the return type of the decorated function. Defaults to object, if not specified or to str if cast was not possible.
    """
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
    """"General decorator for the neural processing engine.
    This method is used to decorate functions which can build any expression without examples.

    Args:
        prompt (str): The prompt describing the task. Defaults to 'Summarize the content of the following text:\n'.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of results to be returned, if more are obtained.
        default (object, optional): Default value if prediction fails. Defaults to None.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        **decorator_kwargs: Additional arguments as key-value pairs passed to the decorated function, which can later accessed in pre_processors and post_processors via the argument.kwargs['key'] dictionary.

    Returns:
        object: The prediction of the model based on the return type of the decorated function. Defaults to object, if not specified or to str if cast was not possible.
    """
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
    """General decorator for the neural processing engine.

    Args:
        message (str): The prompt message describing the task.
        **decorator_kwargs: Additional arguments as key-value pairs passed to the decorated function, which can later accessed in pre_processors and post_processors via the argument.kwargs['key'] dictionary.

    Returns:
        object: The prediction of the model based on the return type of the decorated function. Defaults to object, if not specified or to str if cast was not possible.
    """
    return few_shot(processed_input=message,
                    raw_input=True,
                    **decorator_kwargs)


def summarize(prompt: str = 'Summarize the content of the following text:\n',
              context: Optional[str] = None,
              constraints: List[Callable] = [],
              default: Optional[object] = None,
              pre_processors: Optional[List[pre.PreProcessor]] = [pre.SummaryPreProcessing()],
              post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
              **decorator_kwargs):
    """Summarizes the content of a text.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Summarize the content of the following text:\n'.
        context (str, optional): Provide the context how text should be summarized. Defaults to None.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The summary of the text.
    """
    return few_shot(prompt,
                    context=context,
                    examples=[],
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def equals(context: str = 'contextually',
           default: bool = False,
           prompt: str = "Make a fuzzy equals comparison. Are the following objects {} the same?\n",
           examples: prm.Prompt = prm.FuzzyEquals(),
           constraints: List[Callable] = [],
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.EqualsPreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """Equality function for two objects.

    Args:
        context (str, optional): Keyword to express how to compare the words. Defaults to 'contextually'. As an alternative, one can use other type such as 'literally'.
        default (bool, optional): Condition outcome. Defaults to False.
        prompt (str, optional): The prompt describing the task. Defaults to "Are the following objects {} the same?\n".
        examples (Prompt, optional): List of fuzzy examples showing how to compare objects in specified format. Defaults to FuzzyEquals().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [EqualsPreProcessor()] and uses 'self' plus one required argument for comparison (other).
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        bool: The equality of the two objects.
    """
    return few_shot(prompt=prompt.format(context),
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    max_tokens=10,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def sufficient(query: str,
               prompt: str = "Consider if there is sufficient information to answer the query\n",
               default: bool = False,
               examples: prm.Prompt = prm.SufficientInformation(),
               constraints: List[Callable] = [],
               pre_processors: Optional[List[pre.PreProcessor]] = [pre.SufficientInformationPreProcessor()],
               post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
               **decorator_kwargs) -> bool:
    """Determines if there is sufficient information to answer the given query.

    Args:
        query (str): The query to be evaluated.
        prompt (str, optional): The prompt describing the task. Defaults to "Consider if there is sufficient information to answer the query"
        default (bool, optional): The default value to be returned if the task cannot be solved. Defaults to False. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of example inputs used to train the model. Defaults to SufficientInformation().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [SufficientInformationPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        bool: True if there is sufficient information to answer the query, False otherwise.
    """
    return few_shot(prompt=prompt,
                    query=query,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    max_tokens=10,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def delitem(default: Optional[str] = None,
            prompt: str = "Delete the items at the index position\n",
            examples: prm.Prompt = prm.RemoveIndex(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.DeleteIndexPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Deletes the items at the specified index position.

    Args:
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to 'Delete the items at the index position'
        examples (Prompt, optional): A list of strings from which the model can learn. Defaults to RemoveIndex().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [DeleteIndexPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The item at the specified index position.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def setitem(default: Optional[str] = None,
            prompt: str = "Set item at index position\n",
            examples: prm.Prompt = prm.SetIndex(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.SetIndexPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Sets an item at a given index position in a sequence.

    Args:
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to "Set item at index position"
        examples (Prompt, optional): A list of examples that the model should be trained on. Defaults to SetIndex().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [SetIndexPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The item set at the specified index position.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def getitem(default: Optional[str] = None,
            prompt: str = "Get item at index position\n",
            examples: prm.Prompt = prm.Index(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.IndexPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Retrieves the item at the given index position.

    Args:
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to 'Get item at index position
        examples (Prompt, optional): A list of examples to be used for training. Defaults to Index().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [IndexPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The item at the given index position.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def modify(changes: str,
           default: Optional[str] = None,
           prompt: str = "Modify the text to match the criteria:\n",
           examples: prm.Prompt = prm.Modify(),
           constraints: List[Callable] = [],
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.ModifyPreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """A function to modify a text based on a set of criteria.

    Args:
        changes (str): The criteria to modify the text.
        default (str, optional): A default result if specified. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to "Modify the text to match the criteria:\n".
        examples (Prompt, optional): List of possible modifications in specified format. Defaults to Modify().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ModifyPreProcessor()] and requires one argument (text).
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The modified text.
    """
    return few_shot(prompt=prompt,
                    changes=changes,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def filtering(criteria: str,
              include: bool = False,
              default: Optional[str] = None,
              prompt: str = "Filter the information from the text based on the filter criteria. Leave sentences unchanged if they are unrelated to the filter criteria:\n",
              examples: prm.Prompt = prm.Filter(),
              constraints: List[Callable] = [],
              pre_processors: Optional[List[pre.PreProcessor]] = [pre.FilterPreProcessor()],
              post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
              **decorator_kwargs):
    """Filter information from a text based on a set of criteria.

    Args:
        criteria (str): A description of the criteria to filter the text.
        include (bool, optional): If True, include the information matching the criteria. Defaults to False.
        default (str, optional): A default result if specified. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to "Remove the information from the text based on the filter criteria:\n".
        examples (Prompt, optional): List of filtered examples in specified format. Defaults to Filter().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [FilterPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The filtered text.
    """
    return few_shot(prompt=prompt,
                    criteria=criteria,
                    include=include,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def notify(subscriber: Dict[str, Callable],
           default: Optional[object] = None,
           prompt: str = "List the semantically related topics:\n",
           examples: prm.Prompt = prm.SemanticMapping(),
           constraints: List[Callable] = [],
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.SemanticMappingPreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.SplitPipePostProcessor(), post.NotifySubscriberPostProcessor()],
           **decorator_kwargs):
    """Notify subscribers based on a set of topics if detected in the input text and matching the key of the subscriber.

    Args:
        subscriber (Dict[str, Callable], optional): Dictionary of key-value pairs, with the key being the topic and the value being the function to be called if the topic is detected in the input text.
        default (object, optional): A default result if specified. Defaults to None.
        prompt (_type_, optional): The prompt describing the task. Defaults to "List the semantically related topics:\n".
        examples (Prompt, optional): List of semantic mapping examples. Defaults to SemanticMapping().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [SemanticMappingPreProcessor()]. Requires one argument (text).
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [SplitPipePostProcessor(), NotifySubscriberPostProcessor()].

    Returns:
        str: A string with a list of topics detected in the input text separated by a pipe (|).
    """
    return few_shot(prompt=prompt,
                    subscriber=subscriber,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def compare(default: bool = False,
            operator: str = '>',
            prompt: str = "Compare number 'A' to 'B':\n",
            examples: prm.Prompt = prm.CompareValues(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ComparePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Compare two objects based on the specified operator.

    Args:
        default (bool, optional): The conditional outcome of the comparison. Defaults to False.
        operator (str, optional): A logical operator comparing the two statements. Defaults to '>'.
        prompt (_type_, optional): The prompt describing the task. Defaults to "Compare number 'A' to 'B':\n".
        examples (Prompt, optional): List of comparison examples. Defaults to CompareValues().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [ComparePreProcessor()]. Uses 'self' for 'A' and requires exactly one argument (B) to compare.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        bool: Conditional outcome of the comparison.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    max_tokens=10,
                    operator=operator,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def convert(format: str,
            default: Optional[str] = None,
            prompt: str = "Translate the following text into {} format.\n",
            examples: prm.Prompt = prm.Format(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.TextFormatPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Transformation operation from one format to another.

    Args:
        format (str): Description of how to format the text.
        default (str, optional): A default result if specified. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to "Translate the following text into {} format.\n".
        examples (Prompt, optional): List of format examples. Defaults to Format().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (list, optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [TextFormatPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The formatted text.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    format=format,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def transcribe(modify: str,
               default: Optional[str] = None,
               prompt: str = "Transcribe the following text by only modifying the text by the provided instruction.\n",
               examples: prm.Prompt = prm.Transcription(),
               constraints: List[Callable] = [],
               pre_processors: Optional[List[pre.PreProcessor]] = [pre.TranscriptionPreProcessor()],
               post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
               **decorator_kwargs):
    """Transcription operation of a text to another styled text.

    Args:
        modify (str): Description of how to modify the transcription.
        default (str, optional): A default result if specified. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to "Transcribe the following text by only modifying the text by the provided instruction.\n".
        examples (Prompt, optional): List of format examples. Defaults to Format().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (list, optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [TextFormatPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The transcribed text.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    modify=modify,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def style(description: str,
          libraries: List[str] = [],
          default: Optional[str] = None,
          prompt: str = "Style the [DATA] based on best practices and the descriptions in [...] brackets. Do not remove content from the data! Do not add libraries or other descriptions. \n",
          constraints: List[Callable] = [],
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.StylePreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
          **decorator_kwargs):
    """Styles a given text based on best practices and a given description.

    Args:
        description (str): The description of the style to be applied.
        libraries (List[str], optional): A list of libraries to be used. Defaults to [].
        default (str, optional): The default style to be applied if the task cannot be solved. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to 'Style the following content based on best practices and the following description. Do not change content of the data!
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [StylePreProcessor(), TemplatePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The given text, styled according to best practices.
    """
    return few_shot(prompt=prompt,
                    libraries=libraries,
                    examples=None,
                    constraints=constraints,
                    default=default,
                    limit=1,
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
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ExceptionPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Analyses an Exception and proposes a correction.

    Args:
        query (str): The query of the error.
        exception (Exception): The exception to be analyzed.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to 'Analyses the error and propose a correction.'.
        examples (Prompt, optional): A list of example answers to the error. Defaults to ExceptionMapping().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ExceptionPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The proposed correction for the given error.
    """
    return few_shot(prompt=prompt,
                    query=query,
                    exception=exception,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def correct(context: str,
            exception: Exception,
            default: Optional[str] = None,
            prompt: str = "Correct the code according to the context description.\n",
            examples: Optional[prm.Prompt] = None,
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.CorrectionPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.ConfirmToBoolPostProcessor()],
            **decorator_kwargs):
    """Analyses an Exception and proposes a correction.

    Args:
        context (str): The context of the error.
        exception (Exception): The exception to be analyzed.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to 'Correct the code according to the context description.'.
        examples (Prompt, optional): A list of example answers to the error. Defaults to ExecutionCorrection().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [CorrectionPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The proposed correction for the given error.
    """
    return few_shot(prompt=prompt,
                    context=context,
                    exception=exception,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def translate(language: str = 'English',
              default: str = "Sorry, I do not understand the given language.",
              prompt: str = "Translate the following text into {}:\n",
              examples: Optional[prm.Prompt] = None,
              constraints: List[Callable] = [],
              pre_processors: Optional[List[pre.PreProcessor]] = [pre.LanguagePreProcessor()],
              post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
              **decorator_kwargs):
    """Translates a given text into a specified language.

    Args:
        language (str, optional): The language to which the text should be translated. Defaults to 'English'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to "Sorry, I do not understand the given language.".
        prompt (str, optional): The prompt describing the task. Defaults to "Translate the following text into {}:".
        examples (Prompt, optional): A list of example texts to be used as a reference. Defaults to [].
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [LanguagePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The translated text.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    language=language,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def rank(default: Optional[object] = None,
         order: str = 'desc',
         prompt: str = "Order the list of objects based on their quality measure and oder literal:\n",
         examples: prm.Prompt = prm.RankList(),
         constraints: List[Callable] = [],
         pre_processors: Optional[List[pre.PreProcessor]] = [pre.RankPreProcessor()],
         post_processors: Optional[List[post.PostProcessor]] = [post.ASTPostProcessor()],
         **decorator_kwargs):
    """Ranks a list of objects based on their quality measure and order literal.

    Args:
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        order (str, optional): The order of the objects in the list. Can be either 'desc' (descending) or 'asc' (ascending). Defaults to 'desc'.
        prompt (str, optional): The prompt describing the task. Defaults to "Order the list of objects based on their quality measure and oder literal:".
        examples (Prompt, optional): A list of examples of ordered objects. Defaults to RankList().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [RankPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [ASTPostProcessor()].

    Returns:
        List[str]: The list of objects in the given order.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    order=order,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def replace(prompt: str = "Replace text parts by string pattern.\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.ReplaceText(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ReplacePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Replaces text parts by a given string pattern.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Replace text parts by string pattern.'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        examples (Prompt, optional): A list of examples to be used to train the model. Defaults to ReplaceText().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ReplacePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The replaced text.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def include(prompt: str = "Include information based on description.\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.IncludeText(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.IncludePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Include information from a description.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Include information based on description.'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples containing information to be included. Defaults to IncludeText().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The included information from the description.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def combine(prompt: str = "Add the two data types in a logical way:\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.CombineText(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.CombinePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Combines two data types in a logical way.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Add the two data types in a logical way:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        examples (Prompt, optional): A list of examples to show how the data should be combined. Defaults to CombineText().
        constraints (List[Callable], optional): A list of constraints applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [CombinePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The combined data types.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def negate(prompt: str = "Negate the following statement:\n",
           default: Optional[str] = None,
           examples: prm.Prompt = prm.NegateStatement(),
           constraints: List[Callable] = [],
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.NegatePreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """Negates a given statement.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Negate the following statement:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        examples (Prompt, optional): A list of example statements to be used for training. Defaults to NegateStatement().
        constraints (List[Callable], optional): A list of constraints applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [NegatePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The negated statement.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def contains(default: bool = False,
             prompt: str = "Is semantically the information of 'A' contained in 'B'?\n",
             examples: prm.Prompt = prm.ContainsValue(),
             constraints: List[Callable] = [],
             pre_processors: Optional[List[pre.PreProcessor]] = [pre.ContainsPreProcessor()],
             post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
             **decorator_kwargs):
    """Determines whether a given string contains another string.

    Args:
        default (bool, optional): The default value to be returned if the task cannot be solved. Defaults to False.
        prompt (str, optional): The prompt describing the task. Defaults to 'Is information 'A' contained in 'B'?'
        examples (Prompt, optional): Examples of strings to check if they contain the given string. Defaults to ContainsValue().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ContainsPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        bool: Whether the given string is contained in the provided string.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    max_tokens=10,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def isinstanceof(default: bool = False,
                 prompt: str = "Detect if 'A' isinstanceof 'B':\n",
                 examples: prm.Prompt = prm.IsInstanceOf(),
                 constraints: List[Callable] = [],
                 pre_processors: Optional[List[pre.PreProcessor]] = [pre.IsInstanceOfPreProcessor()],
                 post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
                 **decorator_kwargs):
    """Detects if one object is an instance of another.

    Args:
        default (bool, optional): The default value to be returned if the task cannot be solved. Defaults to False. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to "Detect if 'A' isinstanceof 'B':".
        examples (Prompt, optional): A list of examples used to train the model. Defaults to IsInstanceOf().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to IsInstanceOfPreProcessor().
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        bool: Whether or not the object is an instance of the other.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    max_tokens=10,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def case(enum: List[str],
         default: str,
         prompt: str = "Classify the text according to one of the following categories: ",
         examples: Optional[prm.Prompt] = None,
         stop: List[str] = ['\n'],
         pre_processors: Optional[List[pre.PreProcessor]] = [pre.EnumPreProcessor(), pre.TextMessagePreProcessor(), pre.PredictionMessagePreProcessor()],
         post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.CaseInsensitivePostProcessor()],
         **decorator_kwargs):
    """Classifies a text according to one of the given categories.

    Args:
        enum (List[str]): A list of strings representing the categories to be classified.
        default (str): The default category to be returned if the task cannot be solved.
        examples (Prompt, optional): A list of examples used to train the model.
        stop (List[str], optional): A list of strings that will stop the prompt. Defaults to ['\n'].
        prompt (str, optional): The prompt describing the task. Defaults to "Classify the text according to one of the following categories: ".
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [EnumPreProcessor(), TextMessagePreProcessor(), PredictionMessagePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor(), CaseInsensitivePostProcessor()].

    Returns:
        str: The category the text is classified as.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    default=default,
                    limit=1,
                    stop=stop,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    enum=enum,
                    **decorator_kwargs)


def extract(prompt: str = "Extract a pattern from text:\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.ExtractPattern(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ExtractPatternPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Extracts a pattern from text.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Extract a pattern from text:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples of the pattern to be extracted. Defaults to ExtractPattern().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ExtractPatternPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The extracted pattern.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def expression(prompt: str = "Evaluate the symbolic expressions:\n",
               default: Optional[str] = None,
               constraints: List[Callable] = [],
               pre_processors: Optional[List[pre.PreProcessor]] = [],
               post_processors: Optional[List[post.PostProcessor]] = [post.WolframAlphaPostProcessor()],
               **decorator_kwargs):
    """Evaluates the symbolic expressions.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Evaluate the symbolic expressions:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [SimpleSymbolicExpressionPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The result of the evaluated expression.
    """
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
                                limit=1,
                                constraints=constraints,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def interpret(prompt: str = "Evaluate the symbolic expressions:\n",
              default: Optional[str] = None,
              examples: prm.Prompt = prm.SimpleSymbolicExpression(),
              constraints: List[Callable] = [],
              pre_processors: Optional[List[pre.PreProcessor]] = [pre.InterpretExpressionPreProcessor()],
              post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
              **decorator_kwargs):
    """Evaluates the symbolic expressions by interpreting the semantic meaning.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Evaluate the symbolic expressions:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [SimpleSymbolicExpressionPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The result of the evaluated expression.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def logic(prompt: str = "Evaluate the logic expressions:\n",
          operator: str = 'and',
          default: Optional[str] = None,
          examples: prm.Prompt = prm.LogicExpression(),
          constraints: List[Callable] = [],
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.LogicExpressionPreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
          **decorator_kwargs):
    """Evaluates a logic expression.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Evaluate the logic expressions:'.
        operator (str, optional): The operator used in the expression. Defaults to 'and'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): The list of examples to be tested. Defaults to LogicExpression().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The evaluated expression.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    operator=operator,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def invert(prompt: str = "Invert the logic of the content:\n",
           default: Optional[str] = None,
           examples: prm.Prompt = prm.InvertExpression(),
           constraints: List[Callable] = [],
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.ArrowMessagePreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """Inverts the logic of a statement.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Invert the logic of the content:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples used to train the model. Defaults to InvertExpression().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ArrowMessagePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The logic of the statement inverted.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def simulate(prompt: str = "Simulate the following code:\n",
             default: Optional[str] = None,
             limit: int = None,
             examples: prm.Prompt = prm.SimulateCode(),
             constraints: List[Callable] = [],
             pre_processors: Optional[List[pre.PreProcessor]] = [pre.SimulateCodePreProcessor()],
             post_processors: Optional[List[post.PostProcessor]] = [post.SplitPipePostProcessor(), post.TakeLastPostProcessor()],
             **decorator_kwargs):
    """Simulates code and returns the result.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Simulate the following code:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The number of results to be returned. Defaults to None.
        examples (Prompt, optional): A list of example codes used to train the model. Defaults to SimulateCode().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [SimulateCodePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [SplitPipePostProcessor(), TakeLastPostProcessor()].

    Returns:
        str: The result of the code simulation.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def code(prompt: str = "Generate code that solves the following problems:\n",
         default: Optional[str] = None,
         limit: int = None,
         examples: prm.Prompt = prm.GenerateCode(),
         constraints: List[Callable] = [],
         pre_processors: Optional[List[pre.PreProcessor]] = [pre.GenerateCodePreProcessor()],
         post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
         **decorator_kwargs):
    """Generates code that solves a given problem.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Generate code that solves the following problems:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        limit (int, optional): The maximum amount of code to be generated. Defaults to None.
        examples (Prompt, optional): A list of given examples of code. Defaults to GenerateCode().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to GenerateCodePreProcessor().
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The generated code that solves the given problem.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def outline(prompt: str = "Outline only the essential content as a short list of bullets. Each bullet is in a new line:\n",
            default: List[str] = None,
            limit: int = None,
            examples: prm.Prompt = prm.TextToOutline(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.TextToOutlinePreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.SplitNewLinePostProcessor()],
            **decorator_kwargs):
    """Outlines the essential content as a short list of bullets.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Outline only the essential content as a short list of bullets. Each bullet is in a new line:".
        default (List[str], optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum length of the output. Defaults to None.
        examples (Prompt, optional): The list of examples provided. Defaults to TextToOutline().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor(), SplitNewLinePostProcessor()].

    Returns:
        List[str]: The short list of bullets outlining the essential content.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def unique(prompt: str = "Create a short unique key that captures the essential topic from the following statements and does not collide with the list of keys:\n",
           keys: List[str] = None,
           default: List[str] = None,
           limit: int = None,
           examples: prm.Prompt = prm.UniqueKey(),
           constraints: List[Callable] = [],
           pre_processors: Optional[List[pre.PreProcessor]] = [pre.UniquePreProcessor()],
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
           **decorator_kwargs):
    """Creates a short, unique key that captures the essential topic from the given statements and does not collide with the list of keys.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Create a short unique key that captures the essential topic from the following statements and does not collide with the list of keys:'.
        keys (List[str], optional): A list of keys to check against for uniqueness. Defaults to None.
        default (List[str], optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of keys to return. Defaults to None.
        examples (Prompt, optional): A list of example keys that the unique key should be based on. Defaults to UniqueKey().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [UniquePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        List[str]: The list of unique keys.
    """
    return few_shot(prompt=prompt,
                    keys=keys,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def clean(prompt: str = "Clean up the text from special characters or escape sequences. DO NOT change any words or sentences! Keep original semantics:\n",
          default: List[str] = None,
          limit: int = None,
          examples: prm.Prompt = prm.CleanText(),
          constraints: List[Callable] = [],
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.CleanTextMessagePreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
          **decorator_kwargs):
    """Cleans up a text from special characters and escape sequences.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Clean up the text from special characters or escape sequences:".
        default (List[str], optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of cleaned up words to be returned. Defaults to None.
        examples (Prompt, optional): A list of examples to be used to train the model. Defaults to [CleanText()].
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [CleanTextMessagePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        List[str]: The cleaned up text.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def compose(prompt: str = "Create a coherent text based on the facts listed in the outline:\n",
            default: Optional[str] = None,
            examples: Optional[prm.Prompt] = None,
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.GenerateTextPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Compose a coherent text based on an outline.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Create a coherent text based on an outline:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples that help guide the model to solve the task. Defaults to [].
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [GenerateTextPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The composed text.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def foreach(condition: str,
            apply: str,
            prompt: str = "Iterate over each element and apply operation based on condition:\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.ForEach(),
            constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ForEachPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Applies an operation based on a given condition to each element in a list.

    Args:
        condition (str): The condition to be applied to each element.
        apply (str): The operation to be applied to each element.
        prompt (str, optional): The prompt describing the task. Defaults to "Iterate over each element and apply operation based on condition:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples to be used by the model. Defaults to ForEach().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        List[str]: A list of elements with the applied operation.
    """
    return few_shot(prompt=prompt,
                    condition=condition,
                    apply=apply,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def dictionary(context: str,
               prompt: str = "Map related content together under a common abstract topic. Do not remove content:\n",
               default: Optional[str] = None,
               examples: prm.Prompt = prm.MapContent(),
               constraints: List[Callable] = [],
               pre_processors: Optional[List[pre.PreProcessor]] = [pre.MapPreProcessor()],
               post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.ASTPostProcessor()],
               **decorator_kwargs):
    """Maps related content together under a common abstract topic.

    Args:
        context (str): The text from which the content is to be mapped.
        prompt (str, optional): The prompt describing the task. Defaults to "Map related content together under a common abstract topic. Do not remove content:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of example content to be mapped. Defaults to MapContent().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [MapPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor(), ASTPostProcessor()].

    Returns:
        str: The mapped content of the text.
    """
    return few_shot(prompt=prompt,
                    context=context,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def listing(condition: str,
            prompt: str = "List each element contained in the text or list based on condition:\n",
            default: Optional[str] = None,
            examples: prm.Prompt = prm.ListObjects(),
             constraints: List[Callable] = [],
            pre_processors: Optional[List[pre.PreProcessor]] = [pre.ListPreProcessor()],
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Lists each element contained in the text or list based on the given condition.

    Args:
        condition (str): The condition to filter elements by.
        prompt (str, optional): The prompt describing the task. Defaults to "List each element contained in the text or list based on condition:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples that can be used to validate the output of the model. Defaults to ListObjects().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        List[str]: The list of elements filtered by the given condition.
    """
    return few_shot(prompt=prompt,
                    condition=condition,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def query(context: str,
          prompt: Optional[str] = None,
          examples: Optional[prm.Prompt] = None,
          constraints: List[Callable] = [],
          default: Optional[object] = None,
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.QueryPreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
          **decorator_kwargs):
    """Performs a query given a context.

    Args:
        context (str): The context for the query.
        prompt (str, optional): The prompt describing the task. Defaults to None.
        examples (Prompt, optional): A list of examples to provide to the model. Defaults to [].
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [QueryPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The answer to the query.
    """
    return few_shot(prompt=prompt,
                    context=context,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def expand(prompt: Optional[str] = 'Write a self-contained function (with all imports) to solve a specific user problem task. Label the function with a name that describes the task.',
           examples: Optional[prm.Prompt] = prm.ExpandFunction(),
           constraints: List[Callable] = [],
           default: Optional[object] = None,
           pre_processors: Optional[List[pre.PreProcessor]] = pre.ExpandFunctionPreProcessor(),
           post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor(), post.ExpandFunctionPostProcessor()],
           **decorator_kwargs):
    """Performs a expand command given a context to generate new prompts.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Write a prompt to condition a large language model to perform an action given a user task'.
        examples (Prompt, optional): A list of examples to provide to the model. Defaults to ExpandFunction().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [QueryPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The answer to the query.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=[prm.Prompt.stop_token],
                    pre_processors=pre_processors,
                    post_processors=post_processors,
                    **decorator_kwargs)


def search(query: str,
           constraints: List[Callable] = [],
           default: Optional[object] = None,
           limit: int = 1,
           pre_processors: Optional[List[pre.PreProcessor]] = None,
           post_processors: Optional[List[post.PostProcessor]] = None,
           **decorator_kwargs):
    """Searches for a given query on the internet.

    Args:
        query (str): The query to be searched for.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        limit (int, optional): The maximum number of results to be returned. Defaults to 1.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to None.
        **decorator_kwargs: Additional keyword arguments to be passed to the decorated function.

    Returns:
        object: The search results based on the query.
    """
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
                                limit=limit,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def opening(path: str,
            constraints: List[Callable] = [],
            default: Optional[object] = None,
            limit: int = None,
            pre_processors: Optional[List[pre.PreProcessor]] = None,
            post_processors: Optional[List[post.PostProcessor]] = [post.StripPostProcessor()],
            **decorator_kwargs):
    """Opens a file and applies a given function to it.

    Args:
        path (str): The path of the file to open.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of results to be returned. Defaults to None.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        **decorator_kwargs: Arbitrary keyword arguments to be passed to the function.

    Returns:
        object: The result of applying the given function to the opened file.
    """
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
                                limit=limit,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def embed(entries: List[str],
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.UnwrapListSymbolsPreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = None,
          **decorator_kwargs):
    """Embeds the entries provided in a decorated function.

    Args:
        entries (List[str]): A list of entries that will be embedded in the decorated function.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the entries. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the entries. Defaults to None.
        **decorator_kwargs: Additional keyword arguments to be passed to the decorated function.

    Returns:
        function: A function with the entries embedded.
    """
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
    """Embeds and clusters the input entries.

    Args:
        entries (List[str]): The list of entries to be clustered.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [ClusterPostProcessor()].
        **decorator_kwargs (optional): Additional keyword arguments to be passed to the underlying embedding model.

    Returns:
        List[List[str]]: The list of clustered entries.
    """
    return embed(entries=entries,
                 pre_processors=pre_processors,
                 post_processors=post_processors,
                 **decorator_kwargs)


def draw(operation: str = 'create',
         prompt: str = '',
         pre_processors: Optional[List[pre.PreProcessor]] = [pre.ValuePreProcessor()],
         post_processors: Optional[List[post.PostProcessor]] = None,
         **decorator_kwargs):
    """Draws an image provided in a decorated function.

    Args:
        operation (str, optional): The specific operation to be performed. Defaults to 'create'.
        prompt (str, optional): The prompt describing context of the image generation process.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the entries. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the entries. Defaults to None.
        **decorator_kwargs: Additional keyword arguments to be passed to the decorated function.

    Returns:
        function: A function with the entries embedded.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            decorator_kwargs['operation'] = operation
            decorator_kwargs['prompt']    = prompt
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(
                                engine='imagerendering',
                                instance=instance,
                                func=func,
                                pre_processors=pre_processors,
                                post_processors=post_processors,
                                argument=argument)
        return wrapper
    return decorator


def text_vision(image: Optional[str] = None,
                text: List[str] = None,
                pre_processors: Optional[List[pre.PreProcessor]] = None,
                post_processors: Optional[List[post.PostProcessor]] = None,
                **decorator_kwargs):
    """Performs vision-related associative tasks. Currently limited to CLIP model embeddings.

    Args:
        image (str, optional): The image the task should be performed on. Defaults to None.
        text (List[str], optional): The text describing the task. Defaults to None.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to None.
        **decorator_kwargs: Additional keyword arguments for the decorated method.

    Returns:
        object: The result of the performed task.
    """
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
    """Performs Optical Character Recognition (OCR) on an image.

    Args:
        image (str): The filepath of the image containing the text to be recognized.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the image before performing OCR. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the output of the OCR before returning the result. Defaults to None.
        **decorator_kwargs: Additional keyword arguments to pass to the decorated function.

    Returns:
        str: The text recognized by the OCR.
    """
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
    """Decorates the given function for speech recognition.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'decode'.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to None.
        **decorator_kwargs: Additional keyword arguments.

    Returns:
        Callable: The decorated function.
    """
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
    """Decorates the given function for text to speech synthesis.

    Args:
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to None.
        **decorator_kwargs: Additional keyword arguments.

    Returns:
        Callable: The decorated function.
    """
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
    """Offers an output stream for writing results.

    Args:
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ConsolePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [ConsolePostProcessor()].
        decorator_kwargs (dict, optional): Keyword arguments to be passed to the wrapped function.

    Returns:
        function: The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *signature_args, **signature_kwargs):
            # Construct container object for the arguments and kwargs
            argument = Argument(signature_args, signature_kwargs, decorator_kwargs)
            return EngineRepository.query(engine='output',
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
          limit: int = 1,
          pre_processors: Optional[List[pre.PreProcessor]] = [pre.CrawlPatternPreProcessor()],
          post_processors: Optional[List[post.PostProcessor]] = [],
          **decorator_kwargs):
    """Fetches data from a given URL and applies the provided post-processors.

    Args:
        url (str): The URL of the page to fetch data from.
        pattern (str, optional): A regular expression to match the desired data. Defaults to ''.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of matching items to return. Defaults to 1.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [CrawlPatternPreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [].
        decorator_kwargs (dict, optional): Additional keyword arguments to pass to the decorated function. Defaults to {}.

    Returns:
        object: The matched data.
    """
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
                                limit=limit,
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
    """Prompts for user input and returns the user response through a decorator.

    Args:
        prompt (str): The prompt that will be displayed to the user.
        constraints (List[Callable], optional): A list of constraints applied to the user input. Defaults to [].
        default (object, optional): The default value to be returned if the user input does not pass the constraints. Defaults to None.
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the desired form. Defaults to [ConsolePreProcessor()].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        **decorator_kwargs (dict): Additional keyword arguments to be passed to the decorated function.

    Returns:
        callable: The decorator function that can be used to prompt for user input.
    """
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
    """Executes a given function after applying constraints, pre-processing and post-processing.

    Args:
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [].
        **decorator_kwargs (optional): The additional keyword arguments to be passed to the decorated function.

    Returns:
        Callable: The decorated function that executes the given function after applying constraints, pre-processing and post-processing.
    """
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
    """Query for a given index and returns the result through a decorator.

    Args:
        prompt (Any): The query to be used by the search, add or config of the index.
        operation (str, optional): The operation to be performed on the index. Defaults to 'search'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processors (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [].
        post_processors (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [].
        **decorator_kwargs (optional): The additional keyword arguments to be passed to the decorated function.

    Returns:
        Callable: The decorated function that returns the indexed object after applying constraints, pre-processing and post-processing.
    """
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
    """Decorates a function to forward commands to the engine backends.

    Args:
        engines (List[str], optional): A list of engines to forward the command to. Defaults to ['all'].
        **decorator_kwargs: Additional keyword arguments to be passed to the decorated function.

    Returns:
        Callable: The decorated function.
    """
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
    """Decorates a function to initialize custom engines as backends.

    Args:
        engines (Dict[str], optional): A dictionary of engines to initialize a custom setup.

    Returns:
        Callable: The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, **kwargs):
            return EngineRepository.register(
                    engines=engines,
                    instance=instance,
                    func=func
                )
        return wrapper
    return decorator


def tune(operation: str = 'create',
         pre_processors: Optional[List[pre.PreProcessor]] = None,
         post_processors: Optional[List[post.PostProcessor]] = None,
         **decorator_kwargs):
    """Fine tune a LLM.

    Args:
        operation (str, optional): The specific operation to be performed. Defaults to 'create'.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the entries. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the entries. Defaults to None.
        **decorator_kwargs: Additional keyword arguments to be passed to the decorated function.

    Returns:
        function: A function with the entries embedded.
    """
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
    """Caption the content of an image.

    Args:
        image (str, optional): The path to the image to be captioned.
        prompt (str, optional): The prompt describing context of the image generation process.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the entries. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the entries. Defaults to None.
        **decorator_kwargs: Additional keyword arguments to be passed to the decorated function.

    Returns:
        function: A function with the entries embedded.
    """
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
