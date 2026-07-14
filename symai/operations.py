import ast
import base64
import re
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import TypeVar, cast

from pydantic import BaseModel

from symai.prompts import CompareValues, FuzzyEquals
from symai.runtime.models import (
    EmbeddingRequest,
    ImageContent,
    ImageDetail,
    LanguageModelRequest,
    LanguageModelResponse,
    ResponseMetadata,
    SamplingConfig,
    SystemMessage,
    TextContent,
    UserMessage,
)

ValueT = TypeVar("ValueT")

_EQUALS_EXAMPLES = tuple(FuzzyEquals().value)
_COMPARE_EXAMPLES = tuple(CompareValues().value)
_MEDIA_TYPE_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*"
)


class BooleanMode(StrEnum):
    STRICT = "strict"
    MEDIUM = "medium"
    TOLERANT = "tolerant"


_BOOLEAN_TRUE_VALUES = {
    BooleanMode.STRICT: frozenset({"true"}),
    BooleanMode.MEDIUM: frozenset({"true", "yes", "ok", "['true']"}),
    BooleanMode.TOLERANT: frozenset(
        {
            "true",
            "1",
            "t",
            "y",
            "yes",
            "yeah",
            "yup",
            "certainly",
            "ok",
            "['true']",
        }
    ),
}


class _Missing:
    __slots__ = ()


_MISSING = _Missing()


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


def parse_boolean(value: str, *, mode: BooleanMode = BooleanMode.MEDIUM) -> bool:
    return value.strip().lower() in _BOOLEAN_TRUE_VALUES[mode]


def parse_typed_value(value: str, return_type: type[ValueT]) -> ValueT:
    if return_type is str:
        return cast("ValueT", value)
    if return_type is bool:
        return cast("ValueT", parse_boolean(value))
    if return_type in (list, tuple, set, dict):
        parsed = ast.literal_eval(value)
        if type(parsed) is not return_type:
            msg = f"Expected {return_type.__name__}, received {type(parsed).__name__}"
            raise ValueError(msg)
        return cast("ValueT", parsed)
    if issubclass(return_type, BaseModel):
        return cast("ValueT", return_type.model_validate_json(value))
    converter = cast("Callable[[str], ValueT]", return_type)
    return converter(value)


def parse_typed_output(
    response: LanguageModelResponse,
    return_type: type[ValueT],
    *,
    index: int = 0,
    default: ValueT | _Missing = _MISSING,
    limit: int | None = None,
) -> ValueT:
    text = _output_text(response, index)
    try:
        parsed = parse_typed_value(text, return_type)
    except (SyntaxError, TypeError, ValueError):
        if isinstance(default, _Missing):
            raise
        parsed = default
    return limit_value(parsed, limit)


def parse_typed_output_with_metadata(
    response: LanguageModelResponse,
    return_type: type[ValueT],
    *,
    index: int = 0,
    default: ValueT | _Missing = _MISSING,
    limit: int | None = None,
) -> tuple[ValueT, ResponseMetadata]:
    value = parse_typed_output(
        response,
        return_type,
        index=index,
        default=default,
        limit=limit,
    )
    return value, response.metadata


def limit_value(value: ValueT, limit: int | None) -> ValueT:
    if limit is None:
        return value
    if limit <= 0:
        msg = "limit must be greater than zero"
        raise ValueError(msg)
    if isinstance(value, list):
        return cast("ValueT", value[:limit])
    if isinstance(value, tuple):
        return cast("ValueT", value[:limit])
    if isinstance(value, dict):
        return cast("ValueT", dict(tuple(value.items())[:limit]))
    if isinstance(value, set):
        msg = "Cannot deterministically limit an unordered set"
        raise TypeError(msg)
    return value


def _string_tuple(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, str):
        msg = f"{field} must be a sequence of strings, not one string"
        raise TypeError(msg)
    return tuple(values)


def _output_text(response: LanguageModelResponse, index: int) -> str:
    for output in response.outputs:
        if output.index == index:
            return output.text
    msg = f"Language response did not contain output index {index}"
    raise IndexError(msg)
