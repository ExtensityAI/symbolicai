from collections.abc import Callable
from typing import cast, override

from symai.runtime.models import LanguageModelResponse

_TRUE_VALUES = frozenset({"true", "yes", "1"})
_FALSE_VALUES = frozenset({"false", "no", "0"})


class DecodeError(ValueError):
    """Expected failure to convert model output into a requested value."""


class Missing:
    """Marker type distinguishing an omitted decode default from any real value."""

    __slots__ = ()

    @override
    def __repr__(self) -> str:
        return "MISSING"


MISSING = Missing()


def decode_text(text: str, /) -> str:
    """Strip surrounding whitespace, leaving quotes and inner text untouched.

    Use `response.text` instead when the raw provider text is wanted verbatim.
    """
    return text.strip()


def decode_bool(text: str, /) -> bool:
    """Decode an explicit boolean word, rejecting anything ambiguous.

    Accepts true/yes/1 and false/no/0, case-insensitively and optionally quoted.
    Prose such as "probably" is a decode failure rather than a silent truthy value.
    """
    normalized = _normalize_scalar_text(text).casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    msg = f"Expected an explicit boolean value, received {text!r}"
    raise DecodeError(msg)


def scalar_decoder[T](constructor: Callable[[str], T]) -> Callable[[str], T]:
    """Build a decoder that normalizes scalar text before calling `constructor`.

    Normalization strips whitespace and one layer of surrounding single quotes, so a
    model that answers `'42'` still decodes. Pass `constructor` directly to
    `decode_output` instead when the output is known to be clean; use a `TypeAdapter`
    for containers and models, which need structural parsing rather than a constructor.

    Raises:
        TypeError: if `constructor` is a container type.
    """
    if constructor in (list, tuple, set, frozenset, dict):
        msg = "Container outputs require a TypeAdapter decoder, not a constructor"
        raise TypeError(msg)
    if constructor is bool:
        return cast("Callable[[str], T]", decode_bool)

    def decode(text: str, /) -> T:
        return constructor(_normalize_scalar_text(text))

    return decode


def decode_output[T](
    response: LanguageModelResponse,
    decoder: Callable[[str], T],
    *,
    output_index: int = 0,
    default: T | Missing = MISSING,
    limit: int | None = None,
) -> T:
    """Select one output, decode its text, then apply an optional deterministic limit.

    A decoder is any `Callable[[str], T]` — `int`, `decode_text`, `decode_bool`,
    `scalar_decoder(int)`, or `TypeAdapter(list[User]).validate_json`. Decoders report
    failure by raising SyntaxError/TypeError/ValueError (pydantic's ValidationError is a
    ValueError); those surface as `DecodeError` and are the only failures `default`
    replaces. Output selection, limiting, and every other decoder exception always
    propagate, so a decoder bug is never silently converted into `default`.

    `limit` keeps the first `limit` entries of an ordered collection — a list, a tuple, or
    a dict. Anything else, including a set and any scalar, is returned unchanged rather
    than limited: an unordered collection has no deterministic first `limit` elements, and
    a limit that silently depended on set iteration order would not be reproducible.

    Raises:
        DecodeError: if the decoder rejected the output and no `default` was given.
        IndexError: if `output_index` is absent from the response.
        ValueError: if `limit` is not greater than zero.
    """
    text = response.output_text(output_index)
    try:
        value = decoder(text)
    except DecodeError:
        if isinstance(default, Missing):
            raise
        value = default
    except (SyntaxError, TypeError, ValueError) as error:
        if isinstance(default, Missing):
            raise _decode_error(decoder, error) from error
        value = default

    return _limit_value(value, limit)


def _decode_error(decoder: object, error: Exception) -> DecodeError:
    name = getattr(decoder, "__name__", type(decoder).__name__)
    msg = f"{name} could not decode output: {error}"
    return DecodeError(msg)


def _normalize_scalar_text(text: str) -> str:
    normalized = text.strip()
    if len(normalized) >= 2 and normalized.startswith("'") and normalized.endswith("'"):
        normalized = normalized[1:-1].strip()
    return normalized


def _limit_value[T](value: T, limit: int | None) -> T:
    if limit is None:
        return value
    if limit <= 0:
        msg = "limit must be greater than zero"
        raise ValueError(msg)
    if isinstance(value, (list, tuple)):
        return cast("T", value[:limit])
    if isinstance(value, dict):
        return cast("T", dict(tuple(value.items())[:limit]))

    return value
