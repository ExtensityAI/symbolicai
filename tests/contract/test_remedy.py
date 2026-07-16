from dataclasses import dataclass

from symai.contract.remedy import remediate


@dataclass(frozen=True)
class RetryPolicy:
    tries: int = 2
    delay: float = 0.1
    max_delay: float = 0.15
    jitter: float = 0.0
    backoff: float = 2.0


def validate_positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        msg = "value must be positive"
        raise ValueError(msg)
    return value


def test_remedy_retries_with_bounded_backoff_until_validation_succeeds() -> None:
    prompts: list[tuple[str, tuple[str, ...]]] = []
    generated = iter(("-2", "3"))
    delays: list[float] = []

    outcome = remediate(
        initial_output="-1",
        validate=validate_positive,
        generate=lambda _prompt: generated.__next__(),
        build_prompt=lambda output, errors: prompts.append((output, errors)) or "fix",
        format_error=str,
        retry=RetryPolicy(),
        accumulate_errors=False,
        sleep=delays.append,
        uniform=lambda _lower, _upper: 0.0,
    )

    assert outcome.value == 3
    assert outcome.attempts == 3
    assert outcome.errors == ("value must be positive", "value must be positive")
    assert prompts == [
        ("-1", ("value must be positive",)),
        ("-2", ("value must be positive",)),
    ]
    assert delays == [0.1, 0.15]


def test_remedy_can_accumulate_errors_in_each_correction_prompt() -> None:
    prompts: list[tuple[str, ...]] = []

    outcome = remediate(
        initial_output="bad-1",
        validate=lambda text: (_ for _ in ()).throw(ValueError(text)),
        generate=lambda _prompt: "bad-2",
        build_prompt=lambda _output, errors: prompts.append(errors) or "fix",
        format_error=str,
        retry=RetryPolicy(tries=1),
        accumulate_errors=True,
        sleep=lambda _delay: None,
        uniform=lambda _lower, _upper: 0.0,
    )

    assert outcome.value is None
    assert outcome.attempts == 2
    assert outcome.errors == ("bad-1", "bad-2")
    assert prompts == [("bad-1",)]
