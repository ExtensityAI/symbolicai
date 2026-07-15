from collections.abc import Sequence
from dataclasses import dataclass

from symai.operations import language_request
from symai.runtime.models import LanguageModelRequest, LanguageModelResponse
from symai.runtime.runtime import Runtime


@dataclass(frozen=True, slots=True, init=False)
class Function:
    """Build and execute one provider-neutral language-model request."""

    prompt: str
    examples: tuple[str, ...]
    max_tokens: int | None
    stop: tuple[str, ...]

    def __init__(
        self,
        prompt: str = "",
        examples: Sequence[str] | str | None = None,
        *,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
    ) -> None:
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "examples", _normalize_examples(examples))
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "stop", _normalize_string_sequence(stop, "stop"))

    def request(self, *values: object) -> LanguageModelRequest:
        """Construct a normalized request without performing I/O."""

        return language_request(
            self.prompt,
            " ".join(str(value) for value in values),
            examples=self.examples,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )

    def __call__(
        self,
        runtime: Runtime,
        *values: object,
        engine: str | None = None,
    ) -> LanguageModelResponse:
        """Execute through the explicit runtime and return its normalized response."""

        return runtime.execute(self.request(*values), engine=engine)

    def execute_many(
        self,
        runtime: Runtime,
        inputs: Sequence[Sequence[object]],
        *,
        engine: str | None = None,
    ) -> tuple[LanguageModelResponse, ...]:
        """Execute nested inputs sequentially while preserving their order."""

        if isinstance(inputs, str):
            msg = "inputs must be a sequence of input sequences, not one string"
            raise TypeError(msg)
        for values in inputs:
            if isinstance(values, str):
                msg = "each input must be a sequence of values, not one string"
                raise TypeError(msg)

        return tuple(self(runtime, *values, engine=engine) for values in inputs)



def _normalize_examples(
    examples: Sequence[str] | str | None,
) -> tuple[str, ...]:
    if examples is None:
        return ()
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
