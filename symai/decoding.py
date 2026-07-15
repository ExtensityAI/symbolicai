import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from symai.runtime.models import LanguageModelResponse

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
ModelT = TypeVar("ModelT", bound=BaseModel)

_TRUE_VALUES = frozenset({"true", "yes", "1"})
_FALSE_VALUES = frozenset({"false", "no", "0"})


class DecodeError(ValueError):
    """Expected failure to convert model output into a requested value."""


class Missing:
    """Marker type distinguishing an omitted decode default from any real value."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING = Missing()


class Decoder(Protocol[T_co]):
    """Strategy that converts normalized language-model text to one result type."""

    def decode(self, text: str, /) -> T_co: ...


@dataclass(frozen=True, slots=True)
class TextDecoder:
    def decode(self, text: str, /) -> str:
        return _normalize_text(text)


@dataclass(frozen=True, slots=True)
class ConstructorDecoder(Generic[T]):
    constructor: type[T]

    def decode(self, text: str, /) -> T:
        normalized = _normalize_text(text)
        try:
            if self.constructor is str:
                return cast("T", normalized)
            if self.constructor is bool:
                return cast("T", _decode_boolean(normalized))
            if self.constructor in (list, tuple, set, dict):
                value = ast.literal_eval(normalized)
                if type(value) is not self.constructor:
                    msg = (
                        f"Expected {self.constructor.__name__}, "
                        f"received {type(value).__name__}"
                    )
                    raise ValueError(msg)
                return cast("T", value)

            converter = cast("Callable[[str], T]", self.constructor)
            return converter(normalized)
        except DecodeError:
            raise
        except (SyntaxError, TypeError, ValueError) as error:
            raise _decode_error(self, error) from error


@dataclass(frozen=True, slots=True)
class TypeAdapterDecoder(Generic[T]):
    adapter: TypeAdapter[T]

    def decode(self, text: str, /) -> T:
        try:
            return self.adapter.validate_json(_normalize_text(text))
        except ValidationError as error:
            raise _decode_error(self, error) from error


@dataclass(frozen=True, slots=True)
class PydanticDecoder(Generic[ModelT]):
    model: type[ModelT]

    def decode(self, text: str, /) -> ModelT:
        try:
            return self.model.model_validate_json(_normalize_text(text))
        except ValidationError as error:
            raise _decode_error(self, error) from error


def decode_output(
    response: LanguageModelResponse,
    decoder: Decoder[T],
    *,
    output_index: int = 0,
    default: T | Missing = MISSING,
    limit: int | None = None,
) -> T:
    """Select one output, decode it, then apply an optional deterministic limit."""

    text = _output_text(response, output_index)
    try:
        value = decoder.decode(text)
    except DecodeError:
        if isinstance(default, Missing):
            raise
        value = default

    return _limit_value(value, limit)


def _decode_error(decoder: object, error: Exception) -> DecodeError:
    msg = f"{type(decoder).__name__} could not decode output: {error}"
    return DecodeError(msg)


def _decode_boolean(text: str) -> bool:
    normalized = text.casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    msg = f"Expected an explicit boolean value, received {text!r}"
    raise ValueError(msg)


def _normalize_text(text: str) -> str:
    normalized = text.strip()
    if len(normalized) >= 2 and normalized.startswith("'") and normalized.endswith("'"):
        normalized = normalized[1:-1].strip()
    return normalized


def _output_text(response: LanguageModelResponse, output_index: int) -> str:
    for output in response.outputs:
        if output.index == output_index:
            return output.text
    msg = f"Language response did not contain output index {output_index}"
    raise IndexError(msg)


def _limit_value(value: T, limit: int | None) -> T:
    if limit is None:
        return value
    if limit <= 0:
        msg = "limit must be greater than zero"
        raise ValueError(msg)
    if isinstance(value, list):
        return cast("T", value[:limit])
    if isinstance(value, tuple):
        return cast("T", value[:limit])
    if isinstance(value, dict):
        return cast("T", dict(tuple(value.items())[:limit]))
    return value
