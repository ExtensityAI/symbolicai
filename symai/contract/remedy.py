import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


class RetryPolicy(Protocol):
    @property
    def tries(self) -> int: ...

    @property
    def delay(self) -> float: ...

    @property
    def max_delay(self) -> float: ...

    @property
    def jitter(self) -> float: ...

    @property
    def backoff(self) -> float: ...


@dataclass(frozen=True, slots=True)
class RemedyOutcome[ValueT]:
    value: ValueT | None
    output: str
    attempts: int
    errors: tuple[str, ...]


def remediate[ValueT](
    *,
    initial_output: str,
    validate: Callable[[str], ValueT],
    generate: Callable[[str], str],
    build_prompt: Callable[[str, Sequence[str]], str],
    format_error: Callable[[Exception], str],
    retry: RetryPolicy,
    accumulate_errors: bool,
    sleep: Callable[[float], object] = time.sleep,
    uniform: Callable[[float, float], float] = random.uniform,
) -> RemedyOutcome[ValueT]:
    """Validate and correct an output within one deterministic retry budget."""
    output = initial_output
    errors: list[str] = []

    for attempt in range(retry.tries + 1):
        try:
            value = validate(output)
        except Exception as error:
            errors.append(format_error(error))
        else:
            return RemedyOutcome(
                value=value,
                output=output,
                attempts=attempt + 1,
                errors=tuple(errors),
            )

        if attempt == retry.tries:
            break

        base_delay = retry.delay * retry.backoff**attempt
        delay = min(retry.max_delay, max(0.0, base_delay + uniform(-retry.jitter, retry.jitter)))
        sleep(delay)
        prompt_errors = tuple(errors) if accumulate_errors else (errors[-1],)
        output = generate(build_prompt(output, prompt_errors))

    return RemedyOutcome(
        value=None,
        output=output,
        attempts=retry.tries + 1,
        errors=tuple(errors),
    )
