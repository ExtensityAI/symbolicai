from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from symai.contract.models import LLMDataModel
from symai.contract.remedy import remediate
from symai.contract.validation import (
    build_remedy_prompt,
    check_semantic_conditions,
    parse_output,
    structured_request,
    validation_errors,
)
from symai.runtime.runtime import LanguageModel

type ContractStage = Literal["pre", "post", "type"]
_PRE_STAGE: ContractStage = "pre"
_POST_STAGE: ContractStage = "post"
_TYPE_STAGE: ContractStage = "type"


class ContractViolation(Exception):
    """Raised when a typed contract cannot satisfy one validation stage."""

    def __init__(self, stage: ContractStage, errors: tuple[str, ...]) -> None:
        self.stage = stage
        self.errors = errors
        super().__init__(f"Contract {stage} validation failed: {'; '.join(errors)}")


class RetryParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tries: int = Field(default=8, ge=1)
    delay: float = Field(default=0.015, ge=0, allow_inf_nan=False)
    max_delay: float = Field(default=0.25, ge=0, allow_inf_nan=False)
    jitter: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    backoff: float = Field(default=1.25, ge=1, allow_inf_nan=False)


@dataclass(frozen=True, slots=True)
class ContractOptions:
    pre_remedy: bool = False
    post_remedy: bool = True
    accumulate_errors: bool = False
    retry: RetryParams = field(default_factory=RetryParams)


@dataclass(frozen=True, slots=True)
class ContractResult[OutputT]:
    value: OutputT | None
    succeeded: bool
    attempts: int
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.attempts < 0:
            msg = "Contract attempts cannot be negative"
            raise ValueError(msg)
        if self.succeeded != (self.value is not None):
            msg = "Contract success must agree with the presence of a value"
            raise ValueError(msg)
        if not self.succeeded and not self.errors:
            msg = "A failed contract result must contain at least one error"
            raise ValueError(msg)


def _noop(_value: object) -> None:
    pass


def _identity[ValueT](value: ValueT) -> ValueT:
    return value


@dataclass(frozen=True, slots=True)
class _NoRetry:
    tries: int = 0
    delay: float = 0.0
    max_delay: float = 0.0
    jitter: float = 0.0
    backoff: float = 1.0


@dataclass(frozen=True, slots=True)
class Contract[InputT: LLMDataModel, OutputT: LLMDataModel]:
    """Immutable typed generation and validation specification."""

    instruction: str
    input_type: type[InputT]
    output_type: type[OutputT]
    pre: Callable[[InputT], None] = _noop
    act: Callable[[InputT], LLMDataModel] = _identity
    post: Callable[[OutputT], None] = _noop
    semantic_conditions: tuple[str, ...] = ()
    options: ContractOptions = field(default_factory=ContractOptions)

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            msg = "Contract instruction must be nonempty"
            raise ValueError(msg)
        if not issubclass(self.input_type, LLMDataModel):
            msg = "Contract input_type must be an LLMDataModel subclass"
            raise TypeError(msg)
        if not issubclass(self.output_type, LLMDataModel):
            msg = "Contract output_type must be an LLMDataModel subclass"
            raise TypeError(msg)
        if not all(condition.strip() for condition in self.semantic_conditions):
            msg = "Contract semantic conditions must be nonempty strings"
            raise ValueError(msg)

    def __call__(
        self,
        engine: LanguageModel,
        input_value: InputT,
        *,
        remedy: LanguageModel | None = None,
    ) -> OutputT:
        result, _processed_input, stage = self._execute(engine, input_value, remedy=remedy)
        if result.value is None:
            if stage is None:
                msg = "A failed contract execution must identify its validation stage"
                raise RuntimeError(msg)
            raise ContractViolation(stage, result.errors)
        return result.value

    def run(
        self,
        engine: LanguageModel,
        input_value: InputT,
        *,
        remedy: LanguageModel | None = None,
    ) -> ContractResult[OutputT]:
        """Execute without raising validation failures."""
        result, _processed_input, _stage = self._execute(engine, input_value, remedy=remedy)
        return result

    def _execute(
        self,
        engine: LanguageModel,
        input_value: InputT,
        *,
        remedy: LanguageModel | None,
    ) -> tuple[ContractResult[OutputT], LLMDataModel, ContractStage | None]:
        if not isinstance(input_value, self.input_type):
            msg = f"Contract input must be {self.input_type.__name__}"
            raise TypeError(msg)

        remedy_engine = remedy or engine
        processed_input, pre_errors, pre_attempts = self._prepare_input(
            remedy_engine,
            input_value,
        )
        if processed_input is None:
            return (
                ContractResult(
                    value=None,
                    succeeded=False,
                    attempts=pre_attempts,
                    errors=pre_errors,
                ),
                input_value,
                "pre",
            )

        acted_input = self.act(processed_input)
        if not isinstance(acted_input, LLMDataModel):
            msg = "Contract act must return an LLMDataModel"
            raise TypeError(msg)

        initial_response = engine.execute(
            structured_request(self.instruction, acted_input.render(), self.output_type)
        )
        initial_output = initial_response.outputs[0].text
        last_stage = _TYPE_STAGE

        def validate(output_text: str) -> OutputT:
            nonlocal last_stage
            try:
                output = parse_output(output_text, self.output_type)
            except ValidationError as error:
                last_stage = _TYPE_STAGE
                raise _StageFailure(_TYPE_STAGE, validation_errors(error)) from error
            try:
                self.post(output)
            except Exception as error:
                last_stage = _POST_STAGE
                raise _StageFailure(_POST_STAGE, _exception_errors(error)) from error
            semantic_errors = check_semantic_conditions(
                remedy_engine,
                output,
                self.semantic_conditions,
            )
            if semantic_errors:
                last_stage = _POST_STAGE
                raise _StageFailure(_POST_STAGE, semantic_errors)
            return output

        retry = self.options.retry if self.options.post_remedy else _NoRetry()
        outcome = remediate(
            initial_output=initial_output,
            validate=validate,
            generate=lambda prompt: _generate(remedy_engine, prompt, self.output_type),
            build_prompt=lambda output, errors: build_remedy_prompt(
                instruction=self.instruction,
                input_text=acted_input.render(),
                output_type=self.output_type,
                output_text=output,
                errors=errors,
            ),
            format_error=_format_stage_failure,
            retry=retry,
            accumulate_errors=self.options.accumulate_errors,
        )
        errors = (*pre_errors, *outcome.errors)
        if outcome.value is None:
            return (
                ContractResult(
                    value=None,
                    succeeded=False,
                    attempts=pre_attempts + outcome.attempts,
                    errors=errors,
                ),
                acted_input,
                last_stage,
            )
        return (
            ContractResult(
                value=outcome.value,
                succeeded=True,
                attempts=pre_attempts + outcome.attempts,
                errors=errors,
            ),
            acted_input,
            None,
        )

    def _prepare_input(
        self,
        remedy_engine: LanguageModel,
        input_value: InputT,
    ) -> tuple[InputT | None, tuple[str, ...], int]:
        try:
            self.pre(input_value)
        except Exception as error:
            initial_errors = _exception_errors(error)
        else:
            return input_value, (), 0

        if not self.options.pre_remedy:
            return None, initial_errors, 0

        def validate(output_text: str) -> InputT:
            try:
                parsed = parse_output(output_text, self.input_type)
            except ValidationError as error:
                raise _StageFailure(_TYPE_STAGE, validation_errors(error)) from error
            try:
                self.pre(parsed)
            except Exception as error:
                raise _StageFailure(_PRE_STAGE, _exception_errors(error)) from error
            return parsed

        outcome = remediate(
            initial_output=input_value.model_dump_json(),
            validate=validate,
            generate=lambda prompt: _generate(remedy_engine, prompt, self.input_type),
            build_prompt=lambda output, errors: build_remedy_prompt(
                instruction=self.instruction,
                input_text=input_value.render(),
                output_type=self.input_type,
                output_text=output,
                errors=errors,
            ),
            format_error=_format_stage_failure,
            retry=self.options.retry,
            accumulate_errors=self.options.accumulate_errors,
        )
        return outcome.value, outcome.errors, outcome.attempts - 1


class _StageFailure(Exception):
    def __init__(self, stage: ContractStage, errors: tuple[str, ...]) -> None:
        self.stage = stage
        self.errors = errors
        super().__init__("\n".join(errors))


def _exception_errors(error: Exception) -> tuple[str, ...]:
    message = str(error).strip()
    return (message or type(error).__name__,)


def _format_stage_failure(error: Exception) -> str:
    if not isinstance(error, _StageFailure):
        raise error
    return "\n".join(error.errors)


def _generate[OutputT: LLMDataModel](
    engine: LanguageModel,
    prompt: str,
    output_type: type[OutputT],
) -> str:
    response = engine.execute(structured_request(prompt, "", output_type))
    return response.outputs[0].text
