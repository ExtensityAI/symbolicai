import base64
import re
from collections.abc import Sequence

from symai.prompts import (
    CombineText,
    CompareValues,
    ContainsValue,
    EndsWith,
    ExtractPattern,
    Filter,
    Format,
    FuzzyEquals,
    IncludeText,
    Index,
    InvertExpression,
    IsInstanceOf,
    LogicExpression,
    MapExpression,
    Modify,
    NegateStatement,
    RankList,
    RemoveIndex,
    ReplaceText,
    SetIndex,
    SimpleSymbolicExpression,
    StartsWith,
)
from symai.runtime.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    ImageContent,
    ImageDetail,
    LanguageModelRequest,
    SamplingConfig,
    SystemMessage,
    TextContent,
    UserMessage,
)

_EQUALS_EXAMPLES = tuple(FuzzyEquals().value)
_COMPARE_EXAMPLES = tuple(CompareValues().value)
_CONTAINS_EXAMPLES = tuple(ContainsValue().value)
_STARTSWITH_EXAMPLES = tuple(StartsWith().value)
_ENDSWITH_EXAMPLES = tuple(EndsWith().value)
_ISINSTANCEOF_EXAMPLES = tuple(IsInstanceOf().value)
_INDEX_EXAMPLES = tuple(Index().value)
_SET_INDEX_EXAMPLES = tuple(SetIndex().value)
_REMOVE_INDEX_EXAMPLES = tuple(RemoveIndex().value)
_MODIFY_EXAMPLES = tuple(Modify().value)
_FILTER_EXAMPLES = tuple(Filter().value)
_MAP_EXAMPLES = tuple(MapExpression().value)
_FORMAT_EXAMPLES = tuple(Format().value)
_RANK_EXAMPLES = tuple(RankList().value)
_REPLACE_EXAMPLES = tuple(ReplaceText().value)
_INCLUDE_EXAMPLES = tuple(IncludeText().value)
_COMBINE_EXAMPLES = tuple(CombineText().value)
_EXTRACT_EXAMPLES = tuple(ExtractPattern().value)
_INTERPRET_EXAMPLES = tuple(SimpleSymbolicExpression().value)
_LOGIC_EXAMPLES = tuple(LogicExpression().value)
_NEGATE_EXAMPLES = tuple(NegateStatement().value)
_INVERT_EXAMPLES = tuple(InvertExpression().value)
_MEDIA_TYPE_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*"
)


def language_request(
    system_prompt: str,
    user_prompt: str,
    *,
    examples: Sequence[str] = (),
    max_tokens: int | None = None,
    stop: Sequence[str] = (),
) -> LanguageModelRequest:
    """Build a normalized text request from explicit prompt parts."""
    messages: list[SystemMessage | UserMessage] = []
    system_parts = ((system_prompt,) if system_prompt else ()) + _string_tuple(examples, "examples")
    if system_parts:
        messages.append(SystemMessage(content=(TextContent(text="\n".join(system_parts)),)))
    messages.append(UserMessage(content=(TextContent(text=user_prompt),)))
    return LanguageModelRequest(
        messages=tuple(messages),
        sampling=SamplingConfig(max_tokens=max_tokens, stop=_string_tuple(stop, "stop")),
    )


def contextualize_language_request(
    request: LanguageModelRequest,
    *,
    static_context: str = "",
    dynamic_context: str = "",
) -> LanguageModelRequest:
    """Append explicit Symbol contexts while preserving prompts and examples."""
    sections = []
    if static_context:
        sections.append(f"<STATIC_CONTEXT/>\n{static_context}")
    if dynamic_context:
        sections.append(f"<DYNAMIC_CONTEXT/>\n{dynamic_context}")
    if not sections:
        return request

    context_text = "\n".join(sections)
    messages = list(request.messages)
    for index, message in enumerate(messages):
        if not isinstance(message, SystemMessage):
            continue
        content = message.content
        if len(content) == 1 and isinstance(content[0], TextContent):
            text = f"{content[0].text}\n{context_text}"
            messages[index] = message.model_copy(update={"content": (TextContent(text=text),)})
        else:
            messages[index] = message.model_copy(
                update={"content": (*content, TextContent(text=context_text))}
            )
        break
    else:
        messages.insert(0, SystemMessage(content=(TextContent(text=context_text),)))
    return request.model_copy(update={"messages": tuple(messages)})


def summarize_request(
    text: object,
    *,
    context: str | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] = (),
) -> LanguageModelRequest:
    prefix = f"Context: {context} " if context else ""
    return language_request(
        "Summarize the content of the following text:\n",
        f"{prefix}Text: {text!s}\n",
        max_tokens=max_tokens,
        stop=stop,
    )


def equals_request(
    left: object,
    right: object,
    *,
    context: str = "contextually",
) -> LanguageModelRequest:
    return language_request(
        f"Make a fuzzy equals comparison; are the following objects {context} the same?\n",
        f"{left!s} == {right!s} =>",
        examples=_EQUALS_EXAMPLES,
    )


def compare_request(
    left: object,
    operator: str,
    right: object,
) -> LanguageModelRequest:
    return language_request(
        "Compare 'A' and 'B' based on the operator:\n",
        f"{left!s} {operator} {right!s} =>",
        examples=_COMPARE_EXAMPLES,
    )


def contains_request(container: object, element: object) -> LanguageModelRequest:
    return language_request(
        "Is semantically the information of 'A' contained in 'B'?\n",
        f"{element!s} in {container!s} =>",
        examples=_CONTAINS_EXAMPLES,
    )


def startswith_request(value: object, prefix: object) -> LanguageModelRequest:
    return language_request(
        "Does 'A' start with 'B'?\n",
        f"{value!s} startswith {prefix!s} =>",
        examples=_STARTSWITH_EXAMPLES,
    )


def endswith_request(value: object, suffix: object) -> LanguageModelRequest:
    return language_request(
        "Does 'A' end with 'B'?\n",
        f"{value!s} endswith {suffix!s} =>",
        examples=_ENDSWITH_EXAMPLES,
    )


def isinstanceof_request(value: object, query: object) -> LanguageModelRequest:
    return language_request(
        "Is 'A' an instance of 'B'?\n",
        f"{value!s} isinstanceof {query!s} =>",
        examples=_ISINSTANCEOF_EXAMPLES,
    )


def negate_request(value: object) -> LanguageModelRequest:
    return language_request(
        "Negate the following statement:\n",
        f"{value!s} =>",
        examples=_NEGATE_EXAMPLES,
    )


def invert_request(value: object) -> LanguageModelRequest:
    return language_request(
        "Invert the logic of the content:\n",
        f"{value!s} =>",
        examples=_INVERT_EXAMPLES,
    )


def getitem_request(value: object, key: object) -> LanguageModelRequest:
    return language_request(
        "Get item at index position:\n",
        f"{value!s} index {key!s} =>",
        examples=_INDEX_EXAMPLES,
    )


def setitem_request(
    value: object,
    key: object,
    replacement: object,
    *,
    delete: bool = False,
) -> LanguageModelRequest:
    if delete:
        return language_request(
            "Delete the items at the index position:\n",
            f"{value!s} remove {key!s} =>",
            examples=_REMOVE_INDEX_EXAMPLES,
        )
    return language_request(
        "Set item at index position:\n",
        f"{value!s} index {key!s} set {replacement!s} =>",
        examples=_SET_INDEX_EXAMPLES,
    )


def modify_request(value: object, changes: str) -> LanguageModelRequest:
    return language_request(
        "Modify the text to match the criteria:\n",
        f"text '{value!s}' modify '{changes!s}'=>",
        examples=_MODIFY_EXAMPLES,
    )


def filter_request(value: object, criteria: str, *, include: bool = False) -> LanguageModelRequest:
    operation = "include" if include else "remove"
    return language_request(
        "Filter the information from the text based on the filter criteria. "
        "Leave sentences unchanged if they are unrelated to the filter criteria:\n",
        f"text '{value!s}' {operation} '{criteria!s}' =>",
        examples=_FILTER_EXAMPLES,
    )


def map_request(value: object, instruction: str) -> LanguageModelRequest:
    return language_request(
        "Transform each element in the input based on the instruction. "
        "Preserve container type and elements that don't match the instruction:\n",
        f"text '{value!s}' {instruction!s} =>",
        examples=_MAP_EXAMPLES,
    )


def convert_request(value: object, format: str) -> LanguageModelRequest:
    return language_request(
        f"Translate the following text into {format} format.\n",
        f"text {value!s} format '{format!s}' =>",
        examples=_FORMAT_EXAMPLES,
    )


def style_request(
    value: object,
    description: str,
    *,
    libraries: Sequence[str] = (),
) -> LanguageModelRequest:
    library_text = ", ".join(_string_tuple(libraries, "libraries"))
    return language_request(
        "Style the [DATA] based on best practices and the descriptions in [...] brackets. "
        "Do not remove content from the data! Do not add libraries or other descriptions. \n",
        f"[FORMAT]: {description}\n[LIBRARIES]: {library_text}\n[DATA]:\n{value!s}\n\n",
    )


def translate_request(value: object, language: str = "English") -> LanguageModelRequest:
    return language_request(
        f"Your task is to translate and **only** translate the text into {language}:\n",
        f"{value!s}",
    )


def rank_request(
    value: object,
    measure: object = "alphanumeric",
    *,
    order: str = "desc",
) -> LanguageModelRequest:
    list_value: object = value
    if isinstance(value, str) and "|" in value and "[" not in value:
        list_value = [part.strip() for part in value.split("|") if part.strip()]
    return language_request(
        "Order the list of objects based on their quality measure and oder literal:\n",
        f"order: '{order!s}' measure: '{measure!s}' list: {list_value!s} =>",
        examples=_RANK_EXAMPLES,
    )


def replace_request(value: object, old: object, new: object) -> LanguageModelRequest:
    return language_request(
        "Replace text parts by string pattern.\n",
        f"text '{value!s}' replace '{old!s}' with '{new!s}'=>",
        examples=_REPLACE_EXAMPLES,
    )


def include_request(value: object, information: object) -> LanguageModelRequest:
    return language_request(
        "Include information based on description.\n",
        f"text '{value!s}' include '{information!s}' =>",
        examples=_INCLUDE_EXAMPLES,
    )


def combine_request(left: object, right: object) -> LanguageModelRequest:
    return language_request(
        "Add the two data types in a logical way:\n",
        f"{left!s} + {right!s} =>",
        examples=_COMBINE_EXAMPLES,
    )


def extract_request(value: object, pattern: object) -> LanguageModelRequest:
    return language_request(
        "Extract a pattern from text:\n",
        f"from '{value!s}' extract '{pattern!s}' =>",
        examples=_EXTRACT_EXAMPLES,
    )


def interpret_request(
    value: object,
    *,
    prompt: str = "Evaluate the symbolic expressions and return only the result:\n",
) -> LanguageModelRequest:
    return language_request(prompt, f"{value!s} =>", examples=_INTERPRET_EXAMPLES)


def logic_request(left: object, operator: str, right: object) -> LanguageModelRequest:
    return language_request(
        "Evaluate the logic expressions:\n",
        f"expr :{left!s}: {operator!s} :{right!s}: =>",
        examples=_LOGIC_EXAMPLES,
    )


def query_request(
    value: object,
    context: str,
    *,
    prompt: str | None = None,
    examples: Sequence[str] = (),
) -> LanguageModelRequest:
    return language_request(
        prompt or "",
        f"Data:\n{value!s}\nContext: {context}\nAnswer:",
        examples=examples,
    )


def image_request(
    system_prompt: str,
    user_prompt: str,
    *,
    image_url: str,
    detail: ImageDetail | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] = (),
) -> LanguageModelRequest:
    messages: list[SystemMessage | UserMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=(TextContent(text=system_prompt),)))
    messages.append(
        UserMessage(
            content=(
                TextContent(text=user_prompt),
                ImageContent(url=image_url, detail=detail),
            )
        )
    )
    return LanguageModelRequest(
        messages=tuple(messages),
        sampling=SamplingConfig(max_tokens=max_tokens, stop=_string_tuple(stop, "stop")),
    )


def data_uri(data: bytes, media_type: str) -> str:
    if _MEDIA_TYPE_PATTERN.fullmatch(media_type) is None:
        msg = "media_type must be a valid type/subtype token"
        raise ValueError(msg)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def embedding_request(
    inputs: Sequence[str],
    *,
    dimensions: int | None = None,
    user: str | None = None,
) -> EmbeddingRequest:
    return EmbeddingRequest(
        inputs=_string_tuple(inputs, "inputs"),
        dimensions=dimensions,
        user=user,
    )


def parse_embedding_response(response: EmbeddingResponse) -> list[list[float]]:
    indices = tuple(vector.index for vector in response.vectors)
    if len(indices) != len(set(indices)):
        msg = "Embedding response indices must be unique"
        raise ValueError(msg)
    return [
        [float(value) for value in vector.values]
        for vector in sorted(response.vectors, key=lambda vector: vector.index)
    ]

def _string_tuple(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, str):
        msg = f"{field} must be a sequence of strings, not one string"
        raise TypeError(msg)
    return tuple(values)


