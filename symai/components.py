from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import Literal, overload

from symai.operations import language_request, parse_typed_output
from symai.prompts import Prompt
from symai.runtime.models import LanguageModelRequest, ResponseMetadata  # noqa: TC001
from symai.runtime.runtime import current_runtime
from symai.symbol import Symbol

_NO_DEFAULT = object()


class Function:
    """Execute one typed language operation through the active runtime."""

    def __init__(
        self,
        prompt: str = "",
        examples: Sequence[str] | Prompt | str | None = None,
        *,
        default: object = _NO_DEFAULT,
        return_type: type = str,
        sym_return_type: type[Symbol] = Symbol,
        limit: int | None = 1,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
        static_context: str = "",
        dynamic_context: str = "",
    ) -> None:
        if not isinstance(return_type, type):
            msg = "return_type must be a concrete type"
            raise TypeError(msg)
        if default is not _NO_DEFAULT and not isinstance(default, return_type):
            msg = "default must be an instance of return_type"
            raise TypeError(msg)
        if not isinstance(sym_return_type, type) or not issubclass(sym_return_type, Symbol):
            msg = "sym_return_type must be a Symbol type"
            raise TypeError(msg)
        if limit is not None and limit <= 0:
            msg = "limit must be greater than zero"
            raise ValueError(msg)
        self._prompt_template = prompt
        self._prompt_format_args: tuple[object, ...] = ()
        self._prompt_format_kwargs: dict[str, object] = {}
        self.examples = _normalize_examples(examples)
        self.default = default
        self.return_type = return_type
        self.sym_return_type = sym_return_type
        self.limit = limit
        self.max_tokens = max_tokens
        self.stop = _normalize_string_sequence(stop, "stop")
        self.static_context = static_context
        self.dynamic_context = dynamic_context

    def format(self, *args: object, **kwargs: object) -> None:
        self._prompt_format_args = args
        self._prompt_format_kwargs = kwargs

    @overload
    def __call__(
        self,
        *values: object,
        preview: Literal[True],
        return_metadata: bool = False,
        output_index: int = 0,
    ) -> LanguageModelRequest: ...

    @overload
    def __call__(
        self,
        *values: object,
        preview: Literal[False] = False,
        return_metadata: Literal[True],
        output_index: int = 0,
    ) -> tuple[Symbol, ResponseMetadata]: ...

    @overload
    def __call__(
        self,
        *values: object,
        preview: Literal[False] = False,
        return_metadata: Literal[False] = False,
        output_index: int = 0,
    ) -> Symbol: ...

    def __call__(
        self,
        *values: object,
        preview: bool = False,
        return_metadata: bool = False,
        output_index: int = 0,
    ) -> Symbol | tuple[Symbol, ResponseMetadata] | LanguageModelRequest:
        request = self._request(values)
        if preview:
            return request

        response = current_runtime().execute(request)
        if self.default is _NO_DEFAULT:
            parsed = parse_typed_output(
                response,
                self.return_type,
                index=output_index,
                limit=self.limit,
            )
        else:
            parsed = parse_typed_output(
                response,
                self.return_type,
                index=output_index,
                default=self.default,
                limit=self.limit,
            )
        result = self.sym_return_type(parsed)
        if return_metadata:
            return result, response.metadata
        return result

    @overload
    def batch(
        self,
        inputs: Sequence[object],
        *,
        preview: Literal[True],
        return_metadata: bool = False,
        output_index: int = 0,
    ) -> tuple[LanguageModelRequest, ...]: ...

    @overload
    def batch(
        self,
        inputs: Sequence[object],
        *,
        preview: Literal[False] = False,
        return_metadata: Literal[True],
        output_index: int = 0,
    ) -> tuple[tuple[Symbol, ResponseMetadata], ...]: ...

    @overload
    def batch(
        self,
        inputs: Sequence[object],
        *,
        preview: Literal[False] = False,
        return_metadata: Literal[False] = False,
        output_index: int = 0,
    ) -> tuple[Symbol, ...]: ...

    def batch(
        self,
        inputs: Sequence[object],
        *,
        preview: bool = False,
        return_metadata: bool = False,
        output_index: int = 0,
    ) -> tuple[object, ...]:
        """Execute independent inputs in stable order through the active runtime."""
        if isinstance(inputs, str):
            msg = "batch inputs must be a sequence, not one string"
            raise TypeError(msg)
        return tuple(
            self(
                value,
                preview=preview,
                return_metadata=return_metadata,
                output_index=output_index,
            )
            for value in inputs
        )

    def _request(self, values: Sequence[object]) -> LanguageModelRequest:
        return language_request(
            self._system_prompt(),
            " ".join(str(value) for value in values),
            examples=self.examples,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )

    def _system_prompt(self) -> str:
        prompt = self._prompt_template
        if self._prompt_format_args or self._prompt_format_kwargs:
            prompt = prompt.format(
                *self._prompt_format_args,
                **self._prompt_format_kwargs,
            )
        parts = [prompt]
        if self.static_context:
            parts.append(f"<STATIC_CONTEXT/>\n{self.static_context}")
        if self.dynamic_context:
            parts.append(f"<DYNAMIC_CONTEXT/>\n{self.dynamic_context}")
        return "\n".join(part for part in parts if part)


def _normalize_examples(
    examples: Sequence[str] | Prompt | str | None,
) -> tuple[str, ...]:
    if examples is None:
        return ()
    if isinstance(examples, Prompt):
        return tuple(examples.value)
    if isinstance(examples, str):
        return (examples,)
    return _normalize_string_sequence(examples, "examples")


def _normalize_string_sequence(
    values: Sequence[str],
    field: str,
) -> tuple[str, ...]:
    if isinstance(values, str):
        msg = f"{field} must be a sequence of strings, not one string"
        raise TypeError(msg)
    normalized = tuple(values)
    if not all(isinstance(value, str) for value in normalized):
        msg = f"{field} must contain only strings"
        raise TypeError(msg)
    return normalized
