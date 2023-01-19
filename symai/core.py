import functools
from typing import Dict, List, Callable, Optional
from .prompts import *
from .pre_processors import *
from .post_processors import *
from .functional import few_shot_func, symbolic_func, search_func, crawler_func, userinput_func, execute_func, open_func, output_func, \
    command_func, embed_func, vision_func, ocr_func, speech_func, imagerendering_func, setup_func


_symbolic_expression_engine = None


def few_shot(prompt: str,
             examples: Prompt, 
             constraints: List[Callable] = [],
             default: Optional[object] = None, 
             limit: int = 1,
             pre_processor: Optional[List[PreProcessor]] = None,
             post_processor: Optional[List[PostProcessor]] = None,
             **wrp_kwargs):
    """"General decorator for the neural processing engine.
    This method is used to decorate functions which can build any expression in a examples-based way.
    
    Args:
        prompt (str): The prompt describing the task. Defaults to 'Summarize the content of the following text:\n'.
        examples (List[str]): A list of examples to be used for the task in specified format.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of results to be returned, if more are obtained.
        default (object, optional): Default value if prediction fails. Defaults to None.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model to match the format of the examples. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        **wrp_kwargs: Additional arguments as key-value pairs passed to the decorated function, which can later accessed in pre_processors and post_processors via the wrp_params['key'] dictionary.
        
    Returns:
        object: The prediction of the model based on the return type of the decorated function. Defaults to object, if not specified or to str if cast was not possible.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return few_shot_func(wrp_self, 
                                 func=func, 
                                 prompt=prompt, 
                                 examples=examples, 
                                 constraints=constraints, 
                                 default=default, 
                                 limit=limit,
                                 pre_processor=pre_processor, 
                                 post_processor=post_processor,
                                 wrp_kwargs=wrp_kwargs,
                                 args=args, kwargs=kwargs)
        return wrapper
    return decorator


def zero_shot(prompt: str, 
              constraints: List[Callable] = [], 
              default: Optional[object] = None, 
              limit: int = 1,
              pre_processor: Optional[List[PreProcessor]] = None,
              post_processor: Optional[List[PostProcessor]] = None,
              **wrp_kwargs):
    """"General decorator for the neural processing engine.
    This method is used to decorate functions which can build any expression without examples.
    
    Args:
        prompt (str): The prompt describing the task. Defaults to 'Summarize the content of the following text:\n'.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of results to be returned, if more are obtained.
        default (object, optional): Default value if prediction fails. Defaults to None.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        **wrp_kwargs: Additional arguments as key-value pairs passed to the decorated function, which can later accessed in pre_processors and post_processors via the wrp_params['key'] dictionary.
        
    Returns:
        object: The prediction of the model based on the return type of the decorated function. Defaults to object, if not specified or to str if cast was not possible.
    """
    return few_shot(prompt,
                    examples=[],
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    wrp_kwargs=wrp_kwargs)
    
    
def summarize(prompt: str = 'Summarize the content of the following text:\n', 
              context: Optional[str] = None,
              constraints: List[Callable] = [], 
              default: Optional[object] = None, 
              pre_processor: Optional[List[PreProcessor]] = [SummaryPreProcessing()],
              post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
              **wrp_kwargs):
    """Summarizes the content of a text.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Summarize the content of the following text:\n'.
        context (str, optional): Provide the context how text should be summarized. Defaults to None.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The summary of the text.
    """
    return few_shot(prompt,
                    context=context,
                    examples=[],
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def equals(context: str = 'contextually',
           default: bool = False,
           prompt: str = "Are the following objects {} the same?\n",
           examples: Prompt = FuzzyEquals(),
           constraints: List[Callable] = [],
           pre_processor: Optional[List[PreProcessor]] = [EqualsPreProcessor()],
           post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
           **wrp_kwargs):
    """Equality function for two objects.

    Args:
        context (str, optional): Keyword to express how to compare the words. Defaults to 'contextually'. As an alternative, one can use other type such as 'literally'.
        default (bool, optional): Condition outcome. Defaults to False.
        prompt (str, optional): The prompt describing the task. Defaults to "Are the following objects {} the same?\n".
        examples (Prompt, optional): List of fuzzy examples showing how to compare objects in specified format. Defaults to FuzzyEquals().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [EqualsPreProcessor()] and uses 'self' plus one required argument for comparison (other).
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def sufficient(query: str,
               prompt: str = "Consider if there is sufficient information to answer the query\n",
               default: bool = False,
               examples: Prompt = SufficientInformation(),
               constraints: List[Callable] = [],
               pre_processor: Optional[List[PreProcessor]] = [SufficientInformationPreProcessor()],
               post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
               **wrp_kwargs) -> bool:
    """Determines if there is sufficient information to answer the given query.

    Args:
        query (str): The query to be evaluated.
        prompt (str, optional): The prompt describing the task. Defaults to "Consider if there is sufficient information to answer the query"
        default (bool, optional): The default value to be returned if the task cannot be solved. Defaults to False. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of example inputs used to train the model. Defaults to SufficientInformation().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [SufficientInformationPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def delitem(default: Optional[str] = None,
            prompt: str = "Delete the items at the index position\n",
            examples: Prompt = RemoveIndex(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [DeleteIndexPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Deletes the items at the specified index position.

    Args:
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to 'Delete the items at the index position'
        examples (Prompt, optional): A list of strings from which the model can learn. Defaults to RemoveIndex().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [DeleteIndexPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The item at the specified index position.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def setitem(default: Optional[str] = None,
            prompt: str = "Set item at index position\n",
            examples: Prompt = SetIndex(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [SetIndexPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Sets an item at a given index position in a sequence.

    Args:
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to "Set item at index position"
        examples (Prompt, optional): A list of examples that the model should be trained on. Defaults to SetIndex().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [SetIndexPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The item set at the specified index position.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def getitem(default: Optional[str] = None,
            prompt: str = "Get item at index position\n",
            examples: Prompt = Index(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [IndexPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Retrieves the item at the given index position.

    Args:
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to 'Get item at index position
        examples (Prompt, optional): A list of examples to be used for training. Defaults to Index().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [IndexPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The item at the given index position.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def modify(changes: str,
           default: Optional[str] = None,
           prompt: str = "Modify the text to match the criteria:\n",
           examples: Prompt = Modify(),
           constraints: List[Callable] = [],
           pre_processor: Optional[List[PreProcessor]] = [ModifyPreProcessor()],
           post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
           **wrp_kwargs):
    """A function to modify a text based on a set of criteria.

    Args:
        changes (str): The criteria to modify the text.
        default (str, optional): A default result if specified. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to "Modify the text to match the criteria:\n".
        examples (Prompt, optional): List of possible modifications in specified format. Defaults to Modify().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ModifyPreProcessor()] and requires one argument (text).
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The modified text.
    """
    return few_shot(prompt=prompt,
                    changes=changes,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def filtering(criteria: str,
              include: bool = False,
              default: Optional[str] = None,
              prompt: str = "Filter the information from the text based on the filter criteria:\n",
              examples: Prompt = Filter(),
              constraints: List[Callable] = [],
              pre_processor: Optional[List[PreProcessor]] = [FilterPreProcessor()],
              post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
              **wrp_kwargs):
    """Filter information from a text based on a set of criteria.

    Args:
        criteria (str): A description of the criteria to filter the text.
        include (bool, optional): If True, include the information matching the criteria. Defaults to False.
        default (str, optional): A default result if specified. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to "Remove the information from the text based on the filter criteria:\n".
        examples (Prompt, optional): List of filtered examples in specified format. Defaults to Filter().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [FilterPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def notify(subscriber: Dict[str, Callable],
           default: Optional[object] = None,
           prompt: str = "List the semantically related topics:\n",
           examples: Prompt = SemanticMapping(),
           constraints: List[Callable] = [],
           pre_processor: Optional[List[PreProcessor]] = [SemanticMappingPreProcessor()],
           post_processor: Optional[List[PostProcessor]] = [SplitPipePostProcessor(), NotifySubscriberPostProcessor()],
           **wrp_kwargs):
    """Notify subscribers based on a set of topics if detected in the input text and matching the key of the subscriber.

    Args:
        subscriber (Dict[str, Callable], optional): Dictionary of key-value pairs, with the key being the topic and the value being the function to be called if the topic is detected in the input text.
        default (object, optional): A default result if specified. Defaults to None.
        prompt (_type_, optional): The prompt describing the task. Defaults to "List the semantically related topics:\n".
        examples (Prompt, optional): List of semantic mapping examples. Defaults to SemanticMapping().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [SemanticMappingPreProcessor()]. Requires one argument (text).
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [SplitPipePostProcessor(), NotifySubscriberPostProcessor()].

    Returns:
        str: A string with a list of topics detected in the input text separated by a pipe (|).
    """
    return few_shot(prompt=prompt,
                    subscriber=subscriber,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def compare(default: bool = False,
            operator: str = '>',
            prompt: str = "Compare number 'A' to 'B':\n",
            examples: Prompt = CompareValues(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [ComparePreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Compare two objects based on the specified operator.

    Args:
        default (bool, optional): The conditional outcome of the comparison. Defaults to False.
        operator (str, optional): A logical operator comparing the two statements. Defaults to '>'.
        prompt (_type_, optional): The prompt describing the task. Defaults to "Compare number 'A' to 'B':\n".
        examples (Prompt, optional): List of comparison examples. Defaults to CompareValues().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [ComparePreProcessor()]. Uses 'self' for 'A' and requires exactly one argument (B) to compare.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def convert(format: str,
            default: Optional[str] = None,
            prompt: str = "Translate the following text into {} format.\n",
            examples: Prompt = Format(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [FormatPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Transformation operation from one format to another.

    Args:
        format (str): Description of how to format the text.
        default (str, optional): A default result if specified. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to "Translate the following text into {} format.\n".
        examples (Prompt, optional): List of format examples. Defaults to Format().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (list, optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [FormatPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The formatted text.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    format=format,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def transcribe(modify: str,
               default: Optional[str] = None,
               prompt: str = "Transcribe the following text by only modifying the text by the provided instruction.\n",
               examples: Prompt = Transcription(),
               constraints: List[Callable] = [],
               pre_processor: Optional[List[PreProcessor]] = [TranscriptionPreProcessor()],
               post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
               **wrp_kwargs):
    """Transcription operation of a text to another styled text.

    Args:
        modify (str): Description of how to modify the transcription.
        default (str, optional): A default result if specified. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to "Transcribe the following text by only modifying the text by the provided instruction.\n".
        examples (Prompt, optional): List of format examples. Defaults to Format().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (list, optional): A list of pre-processors to be applied to the input and shape the input to the model.. Defaults to [FormatPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The transcribed text.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    modify=modify,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def style(description: str,
          libraries: List[str] = [],
          default: Optional[str] = None,
          prompt: str = "Style the following content based on best practices and the following description. Do not change content of the data! \n",
          constraints: List[Callable] = [],
          pre_processor: Optional[List[PreProcessor]] = [StylePreProcessor()],
          post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
          **wrp_kwargs):
    """Styles a given text based on best practices and a given description.

    Args:
        description (str): The description of the style to be applied.
        libraries (List[str], optional): A list of libraries to be used. Defaults to [].
        default (str, optional): The default style to be applied if the task cannot be solved. Defaults to None.
        prompt (str, optional): The prompt describing the task. Defaults to 'Style the following content based on best practices and the following description. Do not change content of the data! 
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [StylePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def analyze(query: str,
            exception: Exception,
            default: Optional[str] = None,
            prompt: str = "Analyses the error and propose a correction.\n",
            examples: Prompt = ExceptionMapping(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [ExceptionPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Analyses an Exception and proposes a correction.

    Args:
        query (str): The query of the error.
        exception (Exception): The exception to be analyzed.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to 'Analyses the error and propose a correction.'.
        examples (Prompt, optional): A list of example answers to the error. Defaults to ExceptionMapping().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ExceptionPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def correct(context: str,
            exception: Exception,
            default: Optional[str] = None,
            prompt: str = "Correct an code error by following the context description.\n",
            examples: Prompt = ExecutionCorrection(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [CorrectionPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Analyses an Exception and proposes a correction.

    Args:
        context (str): The context of the error.
        exception (Exception): The exception to be analyzed.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to 'Correct an code error by following the context description.'.
        examples (Prompt, optional): A list of example answers to the error. Defaults to ExecutionCorrection().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [CorrectionPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)

    
def translate(language: str = 'English',
              default: str = "Sorry, I do not understand the given language.",
              prompt: str = "Translate the following text into {}:\n",
              examples: Prompt = [],
              constraints: List[Callable] = [],
              pre_processor: Optional[List[PreProcessor]] = [LanguagePreProcessor()],
              post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
              **wrp_kwargs):
    """Translates a given text into a specified language.

    Args:
        language (str, optional): The language to which the text should be translated. Defaults to 'English'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to "Sorry, I do not understand the given language.".
        prompt (str, optional): The prompt describing the task. Defaults to "Translate the following text into {}:".
        examples (Prompt, optional): A list of example texts to be used as a reference. Defaults to [].
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [LanguagePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The translated text.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    language=language,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def rank(default: Optional[object] = None,
         order: str = 'desc',
         prompt: str = "Order the list of objects based on their quality measure and oder literal:\n",
         examples: Prompt = RankList(),
         constraints: List[Callable] = [],
         pre_processor: Optional[List[PreProcessor]] = [RankPreProcessor()],
         post_processor: Optional[List[PostProcessor]] = [ASTPostProcessor()],
         **wrp_kwargs):
    """Ranks a list of objects based on their quality measure and order literal.

    Args:
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        order (str, optional): The order of the objects in the list. Can be either 'desc' (descending) or 'asc' (ascending). Defaults to 'desc'.
        prompt (str, optional): The prompt describing the task. Defaults to "Order the list of objects based on their quality measure and oder literal:".
        examples (Prompt, optional): A list of examples of ordered objects. Defaults to RankList().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [RankPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [ASTPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def replace(prompt: str = "Replace text parts by string pattern.\n",
            default: Optional[str] = None,
            examples: Prompt = ReplaceText(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [ReplacePreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Replaces text parts by a given string pattern.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Replace text parts by string pattern.'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        examples (Prompt, optional): A list of examples to be used to train the model. Defaults to ReplaceText().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ReplacePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The replaced text.
    """
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def include(prompt: str = "Include information based on description.\n",
            default: Optional[str] = None,
            examples: Prompt = IncludeText(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [IncludePreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Include information from a description.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Include information based on description.'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples containing information to be included. Defaults to IncludeText().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The included information from the description.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def combine(prompt: str = "Add the two data types in a logical way:\n",
            default: Optional[str] = None,
            examples: Prompt = CombineText(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [CombinePreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Combines two data types in a logical way.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Add the two data types in a logical way:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        examples (Prompt, optional): A list of examples to show how the data should be combined. Defaults to CombineText().
        constraints (List[Callable], optional): A list of constraints applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [CombinePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The combined data types.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def template(template: str,
             prompt: str = 'Insert the data into the template in the best suitable format (header, tables, paragraphs, buttons, etc.):\n',
             placeholder: str = '{{placeholder}}',
             default: Optional[str] = None,
             examples: Prompt = [],
             constraints: List[Callable] = [],
             pre_processor: Optional[List[PreProcessor]] = [DataTemplatePreProcessor(), TemplatePreProcessor()],
             post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
             **wrp_kwargs):
    """Fills in a template with the given data.

    Args:
        template (str): The template string.
        prompt (str, optional): The prompt describing the task. Defaults to 'Insert the data into the template in the best suitable format (header, tables, paragraphs, buttons, etc.):'.
        placeholder (str, optional): The placeholder string. Defaults to '{{placeholder}}'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples to train the model. Defaults to [].
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [DataTemplatePreProcessor(), TemplatePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The filled template.
    """
    return few_shot(prompt=prompt,
                    template=template,
                    placeholder=placeholder,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def negate(prompt: str = "Negate the following statement:\n",
           default: Optional[str] = None,
           examples: Prompt = NegateStatement(),
           constraints: List[Callable] = [],
           pre_processor: Optional[List[PreProcessor]] = [NegatePreProcessor()],
           post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
           **wrp_kwargs):
    """Negates a given statement.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Negate the following statement:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        examples (Prompt, optional): A list of example statements to be used for training. Defaults to NegateStatement().
        constraints (List[Callable], optional): A list of constraints applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [NegatePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The negated statement.
    """
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def contains(default: bool = False,
             prompt: str = "Is information 'A' contained in 'B'?\n",
             examples: Prompt = ContainsValue(),
             constraints: List[Callable] = [],
             pre_processor: Optional[List[PreProcessor]] = [ContainsPreProcessor()],
             post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
             **wrp_kwargs):
    """Determines whether a given string contains another string.

    Args:
        default (bool, optional): The default value to be returned if the task cannot be solved. Defaults to False.
        prompt (str, optional): The prompt describing the task. Defaults to 'Is information 'A' contained in 'B'?'
        examples (Prompt, optional): Examples of strings to check if they contain the given string. Defaults to ContainsValue().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ContainsPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def isinstanceof(default: bool = False,
                 prompt: str = "Detect if 'A' isinstanceof 'B':\n",
                 examples: Prompt = IsInstanceOf(),
                 constraints: List[Callable] = [],
                 pre_processor: Optional[List[PreProcessor]] = [IsInstanceOfPreProcessor()],
                 post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
                 **wrp_kwargs):
    """Detects if one object is an instance of another.

    Args:
        default (bool, optional): The default value to be returned if the task cannot be solved. Defaults to False. Alternatively, one can implement the decorated function.
        prompt (str, optional): The prompt describing the task. Defaults to "Detect if 'A' isinstanceof 'B':".
        examples (Prompt, optional): A list of examples used to train the model. Defaults to IsInstanceOf().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to IsInstanceOfPreProcessor().
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def case(enum: List[str],
         default: str,
         prompt: str = "Classify the text according to one of the following categories: ",
         examples: Prompt = [],
         stop: List[str] = ['\n'],
         pre_processor: Optional[List[PreProcessor]] = [EnumPreProcessor(), TextMessagePreProcessor(), PredictionMessagePreProcessor()],
         post_processor: Optional[List[PostProcessor]] = [StripPostProcessor(), CaseInsensitivePostProcessor()],
         **wrp_kwargs):
    """Classifies a text according to one of the given categories.

    Args:
        enum (List[str]): A list of strings representing the categories to be classified.
        default (str): The default category to be returned if the task cannot be solved.
        examples (Prompt, optional): A list of examples used to train the model. 
        stop (List[str], optional): A list of strings that will stop the prompt. Defaults to ['\n'].
        prompt (str, optional): The prompt describing the task. Defaults to "Classify the text according to one of the following categories: ".
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [EnumPreProcessor(), TextMessagePreProcessor(), PredictionMessagePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor(), CaseInsensitivePostProcessor()].

    Returns:
        str: The category the text is classified as.
    """
    return few_shot(prompt=prompt,
                    examples=examples,
                    default=default,
                    limit=1,
                    stop=stop,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    enum=enum,
                    **wrp_kwargs)


def extract(prompt: str = "Extract a pattern from text:\n",
            default: Optional[str] = None,
            examples: Prompt = ExtractPattern(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [ExtractPatternPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Extracts a pattern from text.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Extract a pattern from text:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples of the pattern to be extracted. Defaults to ExtractPattern().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ExtractPatternPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The extracted pattern.
    """
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def expression(prompt: str = "Evaluate the symbolic expressions:\n",
               default: Optional[str] = None,
               examples: Prompt = SimpleSymbolicExpression(),
               constraints: List[Callable] = [],
               pre_processor: Optional[List[PreProcessor]] = [SimpleSymbolicExpressionPreProcessor()],
               post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
               eval_engine: Optional[str] = None,
               **wrp_kwargs):
    """Evaluates the symbolic expressions.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Evaluate the symbolic expressions:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples used to train the model. Defaults to SimpleSymbolicExpression().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [SimpleSymbolicExpressionPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        eval_engine (str, optional): The symbolic engine to be used. Defaults to None. Alternatively, one can set the symbolic engine using the command(expression_engine='wolframalpha').

    Returns:
        str: The result of the evaluated expression.
    """
    # either symbolic_engine is set or symbolic_expression_engine is set
    if eval_engine is not None and 'wolframalpha' in eval_engine \
        or _symbolic_expression_engine is not None and 'wolframalpha' in _symbolic_expression_engine:
        # send the expression to wolframalpha
        def decorator(func):
            @functools.wraps(func)
            def wrapper(wrp_self, *args, **kwargs):
                return symbolic_func(wrp_self, 
                                     func=func, 
                                     prompt=prompt,
                                     default=default,
                                     limit=1,
                                     examples=examples,
                                     pre_processor=[WolframAlphaPreProcessor()], # no need for pre-processing since the expression is sent to wolframalpha
                                     post_processor=[WolframAlphaPostProcessor()],
                                     wrp_kwargs=wrp_kwargs,
                                     args=args, kwargs=kwargs)
            return wrapper
        return decorator
    # otherwise, use the default symbolic expression engine
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def logic(prompt: str = "Evaluate the logic expressions:\n",
          operator: str = 'and',
          default: Optional[str] = None,
          examples: Prompt = LogicExpression(),
          constraints: List[Callable] = [],
          pre_processor: Optional[List[PreProcessor]] = [LogicExpressionPreProcessor()],
          post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
          **wrp_kwargs):
    """Evaluates a logic expression.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Evaluate the logic expressions:'.
        operator (str, optional): The operator used in the expression. Defaults to 'and'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): The list of examples to be tested. Defaults to LogicExpression().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    

def invert(prompt: str = "Invert the logic of the content:\n",
           default: Optional[str] = None,
           examples: Prompt = InvertExpression(),
           constraints: List[Callable] = [],
           pre_processor: Optional[List[PreProcessor]] = [ArrowMessagePreProcessor()],
           post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
           **wrp_kwargs):
    """Inverts the logic of a statement.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Invert the logic of the content:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples used to train the model. Defaults to InvertExpression().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ArrowMessagePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The logic of the statement inverted.
    """
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    stop=['\n'],
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def simulate(prompt: str = "Simulate the following code:\n",
             default: Optional[str] = None,
             limit: int = None,
             examples: Prompt = SimulateCode(),
             constraints: List[Callable] = [],
             pre_processor: Optional[List[PreProcessor]] = [SimulateCodePreProcessor()],
             post_processor: Optional[List[PostProcessor]] = [SplitPipePostProcessor(), TakeLastPostProcessor()],
             **wrp_kwargs):
    """Simulates code and returns the result.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Simulate the following code:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The number of results to be returned. Defaults to None.
        examples (Prompt, optional): A list of example codes used to train the model. Defaults to SimulateCode().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [SimulateCodePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [SplitPipePostProcessor(), TakeLastPostProcessor()].

    Returns:
        str: The result of the code simulation.
    """
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)


def code(prompt: str = "Generate code that solves the following problems:\n",
         default: Optional[str] = None,
         limit: int = None,
         examples: Prompt = GenerateCode(),
         constraints: List[Callable] = [],
         pre_processor: Optional[List[PreProcessor]] = [GenerateCodePreProcessor()],
         post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
         **wrp_kwargs):
    """Generates code that solves a given problem.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Generate code that solves the following problems:'.
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        limit (int, optional): The maximum amount of code to be generated. Defaults to None.
        examples (Prompt, optional): A list of given examples of code. Defaults to GenerateCode().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to GenerateCodePreProcessor().
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The generated code that solves the given problem.
    """
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
    
    
def outline(prompt: str = "Outline only the essential content as a short list of bullets. Each bullet is in a new line:\n",
            default: List[str] = None,
            limit: int = None,
            examples: Prompt = TextToOutline(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [TextToOutlinePreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor(), SplitNewLinePostProcessor()],
            **wrp_kwargs):
    """Outlines the essential content as a short list of bullets.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Outline only the essential content as a short list of bullets. Each bullet is in a new line:".
        default (List[str], optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum length of the output. Defaults to None.
        examples (Prompt, optional): The list of examples provided. Defaults to TextToOutline().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor(), SplitNewLinePostProcessor()].

    Returns:
        List[str]: The short list of bullets outlining the essential content.
    """
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    wrp_kwargs=wrp_kwargs)
    
    
def unique(prompt: str = "Create a short unique key that captures the essential topic from the following statements and does not collide with the list of keys:\n",
           keys: List[str] = None,
           default: List[str] = None,
           limit: int = None,
           examples: Prompt = UniqueKey(),
           constraints: List[Callable] = [],
           pre_processor: Optional[List[PreProcessor]] = [UniquePreProcessor()],
           post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
           **wrp_kwargs):
    """Creates a short, unique key that captures the essential topic from the given statements and does not collide with the list of keys.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Create a short unique key that captures the essential topic from the following statements and does not collide with the list of keys:'.
        keys (List[str], optional): A list of keys to check against for uniqueness. Defaults to None.
        default (List[str], optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of keys to return. Defaults to None.
        examples (Prompt, optional): A list of example keys that the unique key should be based on. Defaults to UniqueKey().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [UniquePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        List[str]: The list of unique keys.
    """
    return few_shot(prompt=prompt, 
                    keys=keys,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    wrp_kwargs=wrp_kwargs)
    
    
def clean(prompt: str = "Clean up the text from special characters or escape sequences:\n",
          default: List[str] = None,
          limit: int = None,
          examples: Prompt = CleanText(),
          constraints: List[Callable] = [],
          pre_processor: Optional[List[PreProcessor]] = [CleanTextMessagePreProcessor()],
          post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
          **wrp_kwargs):
    """Cleans up a text from special characters and escape sequences.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Clean up the text from special characters or escape sequences:".
        default (List[str], optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of cleaned up words to be returned. Defaults to None.
        examples (Prompt, optional): A list of examples to be used to train the model. Defaults to [CleanText()].
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [CleanTextMessagePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        List[str]: The cleaned up text.
    """
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=limit,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    wrp_kwargs=wrp_kwargs)
    
    
def compose(prompt: str = "Create a coherent text based on an outline:\n",
            default: Optional[str] = None,
            examples: Prompt = [],
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [GenerateTextPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Compose a coherent text based on an outline.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to "Create a coherent text based on an outline:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples that help guide the model to solve the task. Defaults to [].
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [GenerateTextPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The composed text.
    """
    return few_shot(prompt=prompt, 
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    wrp_kwargs=wrp_kwargs)
    
    
def foreach(condition: str,
            apply: str,
            prompt: str = "Iterate over each element and apply operation based on condition:\n",
            default: Optional[str] = None,
            examples: Prompt = ForEach(),
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [ForEachPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Applies an operation based on a given condition to each element in a list.

    Args:
        condition (str): The condition to be applied to each element.
        apply (str): The operation to be applied to each element.
        prompt (str, optional): The prompt describing the task. Defaults to "Iterate over each element and apply operation based on condition:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples to be used by the model. Defaults to ForEach().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

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
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    wrp_kwargs=wrp_kwargs)
    
    
def dictionary(context: str, 
               prompt: str = "Map related content together under a common abstract topic. Do not remove content:\n",
               default: Optional[str] = None,
               examples: Prompt = MapContent(),
               constraints: List[Callable] = [],
               pre_processor: Optional[List[PreProcessor]] = [MapPreProcessor()],
               post_processor: Optional[List[PostProcessor]] = [StripPostProcessor(), ASTPostProcessor()],
               **wrp_kwargs):
    """Maps related content together under a common abstract topic.

    Args:
        context (str): The text from which the content is to be mapped.
        prompt (str, optional): The prompt describing the task. Defaults to "Map related content together under a common abstract topic. Do not remove content:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of example content to be mapped. Defaults to MapContent().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [MapPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor(), ASTPostProcessor()].

    Returns:
        str: The mapped content of the text.
    """
    return few_shot(prompt=prompt,
                    context=context,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    wrp_kwargs=wrp_kwargs)
    
    
def listing(condition: str,
            prompt: str = "List each element contained in the text or list based on condition:\n",
            default: Optional[str] = None,
            examples: Prompt = ListObjects(),
             constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [ListPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            **wrp_kwargs):
    """Lists each element contained in the text or list based on the given condition.

    Args:
        condition (str): The condition to filter elements by.
        prompt (str, optional): The prompt describing the task. Defaults to "List each element contained in the text or list based on condition:".
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        examples (Prompt, optional): A list of examples that can be used to validate the output of the model. Defaults to ListObjects().
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        List[str]: The list of elements filtered by the given condition.
    """
    return few_shot(prompt=prompt,
                    condition=condition,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    wrp_kwargs=wrp_kwargs)


def query(context: str,
          prompt: Optional[str] = None,
          examples: List[Prompt] = [],
          constraints: List[Callable] = [],
          default: Optional[object] = None, 
          pre_processor: Optional[List[PreProcessor]] = [QueryPreProcessor()],
          post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
          **wrp_kwargs):
    """Performs a query given a context.

    Args:
        context (str): The context for the query.
        prompt (str, optional): The prompt describing the task. Defaults to None.
        examples (List[Prompt], optional): A list of examples to provide to the model. Defaults to [].
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [QueryPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The answer to the query.
    """
    return few_shot(prompt=prompt,
                    context=context,
                    examples=examples,
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    wrp_kwargs=wrp_kwargs)


def search(query: str,
           constraints: List[Callable] = [],
           default: Optional[object] = None, 
           limit: int = 1,
           pre_processor: Optional[List[PreProcessor]] = None,
           post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
           *wrp_args,
           **wrp_kwargs):
    """Searches for a given query on the internet.

    Args:
        query (str): The query to be searched for.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. 
        limit (int, optional): The maximum number of results to be returned. Defaults to 1.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        *wrp_args: Additional arguments to be passed to the decorated function.
        **wrp_kwargs: Additional keyword arguments to be passed to the decorated function.

    Returns:
        object: The search results based on the query.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return search_func(wrp_self,
                               func=func,
                               query=query,
                               constraints=constraints, 
                               default=default, 
                               limit=limit, 
                               pre_processor=pre_processor, 
                               post_processor=post_processor,
                               wrp_args=wrp_args,
                               wrp_kwargs=wrp_kwargs,
                               args=args, kwargs=kwargs)
        return wrapper
    return decorator


def opening(path: str,
            constraints: List[Callable] = [],
            default: Optional[object] = None, 
            limit: int = None,
            pre_processor: Optional[List[PreProcessor]] = None,
            post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
            *wrp_args,
            **wrp_kwargs):
    """Opens a file and applies a given function to it.

    Args:
        path (str): The path of the file to open.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of results to be returned. Defaults to None.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        *wrp_args: Variable length argument list to be passed to the function.
        **wrp_kwargs: Arbitrary keyword arguments to be passed to the function.

    Returns:
        object: The result of applying the given function to the opened file.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return open_func(wrp_self,
                             func=func,
                             path=path,
                             constraints=constraints, 
                             default=default, 
                             limit=limit, 
                             pre_processor=pre_processor, 
                             post_processor=post_processor,
                             wrp_args=wrp_args,
                             wrp_kwargs=wrp_kwargs,
                             args=args, kwargs=kwargs)
        return wrapper
    return decorator


def embed(entries: List[str],
          pre_processor: Optional[List[PreProcessor]] = [UnwrapListSymbolsPreProcessor()],
          post_processor: Optional[List[PostProcessor]] = None,
          *wrp_args,
          **wrp_kwargs):
    """Embeds the entries provided in a decorated function.

    Args:
        entries (List[str]): A list of entries that will be embedded in the decorated function.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the entries. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the entries. Defaults to None.
        *wrp_args: Additional positional arguments to be passed to the decorated function.
        **wrp_kwargs: Additional keyword arguments to be passed to the decorated function.

    Returns:
        function: A function with the entries embedded.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return embed_func(wrp_self,
                              entries=entries,
                              func=func,
                              pre_processor=pre_processor, 
                              post_processor=post_processor,
                              wrp_args=wrp_args,
                              wrp_kwargs=wrp_kwargs,
                              args=args, kwargs=kwargs)
        return wrapper
    return decorator


def cluster(entries: List[str],
            pre_processor: Optional[List[PreProcessor]] = [UnwrapListSymbolsPreProcessor()],
            post_processor: Optional[List[PostProcessor]] = [ClusterPostProcessor()],
            **wrp_kwargs):
    """Embeds and clusters the input entries.

    Args:
        entries (List[str]): The list of entries to be clustered.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [ClusterPostProcessor()].
        **wrp_kwargs (optional): Additional keyword arguments to be passed to the underlying embedding model.

    Returns:
        List[List[str]]: The list of clustered entries.
    """
    return embed(entries=entries,
                 pre_processor=pre_processor,
                 post_processor=post_processor,
                 wrp_kwargs=wrp_kwargs)
    

def draw(operation: str = 'create',
         prompt: str = '',
         pre_processor: Optional[List[PreProcessor]] = [ValuePreProcessor()],
         post_processor: Optional[List[PostProcessor]] = None,
         *wrp_args,
         **wrp_kwargs):
    """Draws an image provided in a decorated function.

    Args:
        operation (str, optional): The specific operation to be performed. Defaults to 'create'.
        prompt (str, optional): The prompt describing context of the image generation process.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the entries. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the entries. Defaults to None.
        *wrp_args: Additional positional arguments to be passed to the decorated function.
        **wrp_kwargs: Additional keyword arguments to be passed to the decorated function.

    Returns:
        function: A function with the entries embedded.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return imagerendering_func(wrp_self,
                                       operation=operation,
                                       prompt=prompt,
                                       func=func,
                                       pre_processor=pre_processor, 
                                       post_processor=post_processor,
                                       wrp_args=wrp_args,
                                       wrp_kwargs=wrp_kwargs,
                                       args=args, kwargs=kwargs)
        return wrapper
    return decorator
    
    
def vision(image: Optional[str] = None,
           text: List[str] = None,
           pre_processor: Optional[List[PreProcessor]] = None,
           post_processor: Optional[List[PostProcessor]] = None,
           *wrp_args,
           **wrp_kwargs):
    """Performs vision-related associative tasks. Currently limited to CLIP model embeddings.

    Args:
        image (str, optional): The image the task should be performed on. Defaults to None.
        text (List[str], optional): The text describing the task. Defaults to None.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to None.
        *wrp_args: Additional positional arguments for the decorated method.
        **wrp_kwargs: Additional keyword arguments for the decorated method.

    Returns:
        object: The result of the performed task.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return vision_func(wrp_self,
                               image=image,
                               prompt=text,
                               func=func,
                               pre_processor=pre_processor, 
                               post_processor=post_processor,
                               wrp_args=wrp_args,
                               wrp_kwargs=wrp_kwargs,
                               args=args, kwargs=kwargs)
        return wrapper
    return decorator


def ocr(image: str,
        pre_processor: Optional[List[PreProcessor]] = None,
        post_processor: Optional[List[PostProcessor]] = None,
        *wrp_args,
        **wrp_kwargs):
    """Performs Optical Character Recognition (OCR) on an image.

    Args:
        image (str): The filepath of the image containing the text to be recognized.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the image before performing OCR. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the output of the OCR before returning the result. Defaults to None.
        *wrp_args: Additional arguments to pass to the decorated function.
        **wrp_kwargs: Additional keyword arguments to pass to the decorated function.

    Returns:
        str: The text recognized by the OCR.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return ocr_func(wrp_self,
                            image=image,
                            func=func,
                            pre_processor=pre_processor, 
                            post_processor=post_processor,
                            wrp_args=wrp_args,
                            wrp_kwargs=wrp_kwargs,
                            args=args, kwargs=kwargs)
        return wrapper
    return decorator


def speech(prompt: str = 'decode',
           pre_processor: Optional[List[PreProcessor]] = None,
           post_processor: Optional[List[PostProcessor]] = None,
           *wrp_args,
           **wrp_kwargs):
    """Decorates the given function for speech recognition.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'decode'.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to None.
        *wrp_args: Additional arguments.
        **wrp_kwargs: Additional keyword arguments.

    Returns:
        Callable: The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return speech_func(wrp_self,
                               prompt=prompt,
                               func=func,
                               pre_processor=pre_processor, 
                               post_processor=post_processor,
                               wrp_args=wrp_args,
                               wrp_kwargs=wrp_kwargs,
                               args=args, kwargs=kwargs)
        return wrapper
    return decorator


def output(constraints: List[Callable] = [],
           default: Optional[object] = None, 
           limit: int = None,
           pre_processor: Optional[List[PreProcessor]] = [ConsolePreProcessor()],
           post_processor: Optional[List[PostProcessor]] = [ConsolePostProcessor()],
           *wrp_args,
           **wrp_kwargs):
    """Offers an output stream for writing results.

    Args:
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of outputs to be printed. Defaults to None.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [ConsolePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [ConsolePostProcessor()].
        wrp_args (tuple, optional): Arguments to be passed to the wrapped function.
        wrp_kwargs (dict, optional): Keyword arguments to be passed to the wrapped function.

    Returns:
        function: The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return output_func(wrp_self,
                               func=func,
                               constraints=constraints, 
                               default=default, 
                               limit=limit, 
                               pre_processor=pre_processor, 
                               post_processor=post_processor,
                               wrp_args=wrp_args,
                               wrp_kwargs=wrp_kwargs,
                               args=args, kwargs=kwargs)
        return wrapper
    return decorator


def fetch(url: str,
          pattern: str = '',
          constraints: List[Callable] = [],
          default: Optional[object] = None, 
          limit: int = 1,
          pre_processor: Optional[List[PreProcessor]] = [CrawlPatternPreProcessor()],
          post_processor: Optional[List[PostProcessor]] = [HtmlGetTextPostProcessor()],
          *wrp_args,
          **wrp_kwargs):
    """Fetches data from a given URL and applies the provided post-processors.

    Args:
        url (str): The URL of the page to fetch data from.
        pattern (str, optional): A regular expression to match the desired data. Defaults to ''.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        limit (int, optional): The maximum number of matching items to return. Defaults to 1.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [CrawlPatternPreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [HtmlGetTextPostProcessor()].
        wrp_args (tuple, optional): Additional arguments to pass to the decorated function. Defaults to ().
        wrp_kwargs (dict, optional): Additional keyword arguments to pass to the decorated function. Defaults to {}.

    Returns:
        object: The matched data.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return crawler_func(wrp_self, 
                                func=func,
                                url=url,
                                pattern=pattern,
                                constraints=constraints, 
                                default=default, 
                                limit=limit, 
                                pre_processor=pre_processor, 
                                post_processor=post_processor,
                                wrp_args=wrp_args,
                                wrp_kwargs=wrp_kwargs,
                                args=args, kwargs=kwargs)
        return wrapper
    return decorator


def userinput(constraints: List[Callable] = [],
              default: Optional[object] = None, 
              pre_processor: Optional[List[PreProcessor]] = [ConsoleInputPreProcessor()],
              post_processor: Optional[List[PostProcessor]] = [StripPostProcessor()],
              *wrp_args,
              **wrp_kwargs):
    """Prompts for user input and returns the user response through a decorator.

    Args:
        prompt (str): The prompt that will be displayed to the user.
        constraints (List[Callable], optional): A list of constraints applied to the user input. Defaults to [].
        default (object, optional): The default value to be returned if the user input does not pass the constraints. Defaults to None.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the desired form. Defaults to [ConsolePreProcessor()].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].
        *wrp_args (tuple): Additional arguments to be passed to the decorated function.
        **wrp_kwargs (dict): Additional keyword arguments to be passed to the decorated function.

    Returns:
        callable: The decorator function that can be used to prompt for user input.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return userinput_func(wrp_self, 
                                  func=func,
                                  constraints=constraints, 
                                  default=default, 
                                  pre_processor=pre_processor, 
                                  post_processor=post_processor,
                                  wrp_args=wrp_args,
                                  wrp_kwargs=wrp_kwargs,
                                  args=args, kwargs=kwargs)
        return wrapper
    return decorator


def execute(default: Optional[str] = None,
            constraints: List[Callable] = [],
            pre_processor: Optional[List[PreProcessor]] = [],
            post_processor: Optional[List[PostProcessor]] = [],
            *wrp_args,
            **wrp_kwargs):
    """Executes a given function after applying constraints, pre-processing and post-processing.

    Args:
        default (str, optional): The default value to be returned if the task cannot be solved. Defaults to None.
        constraints (List[Callable], optional): A list of constrains applied to the model output to verify the output. Defaults to [].
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to [].
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [].
        *wrp_args (optional): The additional arguments to be passed to the decorated function.
        **wrp_kwargs (optional): The additional keyword arguments to be passed to the decorated function.

    Returns:
        Callable: The decorated function that executes the given function after applying constraints, pre-processing and post-processing.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, *args, **kwargs):
            return execute_func(wrp_self, 
                                func=func,
                                code=str(wrp_self),
                                constraints=constraints, 
                                default=default, 
                                pre_processor=pre_processor, 
                                post_processor=post_processor,
                                wrp_args=wrp_args,
                                wrp_kwargs=wrp_kwargs,
                                args=args, kwargs=kwargs)
        return wrapper
    return decorator


def command(engines: List[str] = ['all'], 
            **wrp_kwargs):
    """Decorates a function to forward commands to the engine backends.
    
    Args:
        engines (List[str], optional): A list of engines to forward the command to. Defaults to ['all'].
        wrp_kwargs (dict): A dictionary of keyword arguments to the command function.

    Returns:
        Callable: The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, **kwargs):
            global _symbolic_expression_engine
            if 'symbolic' in engines and 'expression_engine' in wrp_kwargs:
                _symbolic_expression_engine = wrp_kwargs['expression_engine']
            return command_func(wrp_self,
                                func=func,
                                engines=engines,
                                wrp_kwargs=wrp_kwargs,
                                kwargs=kwargs)
        return wrapper
    return decorator


def setup(engines: Dict[str, Any], 
          **wrp_kwargs):
    """Decorates a function to initialize custom engines as backends.
    
    Args:
        engines (Dict[str], optional): A dictionary of engines to initialize a custom setup.
        wrp_kwargs (dict): A dictionary of keyword arguments to the command function.

    Returns:
        Callable: The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(wrp_self, **kwargs):
            return setup_func(wrp_self,
                              func=func,
                              engines=engines,
                              wrp_kwargs=wrp_kwargs,
                              kwargs=kwargs)
        return wrapper
    return decorator
