import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from statistics import mean, pstdev
from time import monotonic
from typing import Any, Protocol, cast

from pydantic import TypeAdapter, ValidationError

from symai.contract.contract import (
    Contract,
    ContractOptions,
    ContractResult,
    ContractStage,
    ContractViolation,
    RetryParams,
)
from symai.contract.models import LLMDataModel, build_dynamic_llm_datamodel
from symai.runtime.runtime import LanguageModel


class _LegacyState(Protocol):
    contract_successful: bool
    contract_result: object | None
    contract_exception: Exception | None


_MISSING = object()


@dataclass(frozen=True, slots=True)
class _DecoratorOptions:
    pre_remedy: bool
    post_remedy: bool
    accumulate_errors: bool
    graceful: bool
    retry: RetryParams
    verbose: bool


@dataclass(frozen=True, slots=True)
class _Timing:
    contract_s: float
    forward_s: float
    attempts: int


def contract(
    *,
    pre_remedy: bool = False,
    post_remedy: bool = False,
    accumulate_errors: bool = False,
    graceful: bool | None = None,
    remedy_retry_params: Mapping[str, object] | RetryParams | None = None,
    verbose: bool = False,
) -> Callable[[type[Any]], type[Any]]:
    """Adapt a legacy class contract to the explicit native contract runtime."""
    options = _decorator_options(
        pre_remedy=pre_remedy,
        post_remedy=post_remedy,
        accumulate_errors=accumulate_errors,
        graceful=graceful,
        remedy_retry_params=remedy_retry_params,
        verbose=verbose,
    )

    def decorate(legacy_type: type[Any]) -> type[Any]:
        original_forward = getattr(legacy_type, "forward", None)
        if not callable(original_forward):
            msg = "A decorated contract class must define forward"
            raise TypeError(msg)
        signature = inspect.signature(original_forward)
        input_parameter = _input_parameter(signature)
        input_model, input_is_dynamic = _model_for_annotation(
            input_parameter.annotation,
            "forward input",
        )
        output_annotation = signature.return_annotation
        output_model, output_is_dynamic = _model_for_annotation(
            output_annotation,
            "forward return",
        )
        output_adapter = TypeAdapter(output_annotation)

        class LegacyContract:
            def __init__(
                self,
                engine: LanguageModel,
                *args: object,
                remedy: LanguageModel | None = None,
                **kwargs: object,
            ) -> None:
                inner = legacy_type(*args, **kwargs)
                prompt = getattr(inner, "prompt", None)
                if not isinstance(prompt, str) or not prompt.strip():
                    msg = "A decorated contract class must define a nonempty prompt"
                    raise TypeError(msg)
                self._inner = inner
                self._engine = engine
                self._remedy = remedy
                self._timings: list[_Timing] = []
                self.contract_successful = False
                self.contract_result: object | None = None
                self.contract_exception: Exception | None = None
                _set_legacy_state(inner, False, None, None)
                self._contract = Contract(
                    instruction=prompt,
                    input_type=input_model,
                    output_type=output_model,
                    pre=_condition(inner, "pre", unwrap=input_is_dynamic),
                    act=_act(inner, input_is_dynamic),
                    post=_condition(inner, "post", unwrap=output_is_dynamic),
                    options=ContractOptions(
                        pre_remedy=options.pre_remedy,
                        post_remedy=options.post_remedy,
                        accumulate_errors=options.accumulate_errors,
                        retry=options.retry,
                    ),
                )

            def __call__(
                self,
                input_value: object = _MISSING,
                *args: object,
                **kwargs: object,
            ) -> object:
                return self.forward(input_value, *args, **kwargs)

            def forward(
                self,
                input_value: object = _MISSING,
                *args: object,
                **kwargs: object,
            ) -> object:
                input_value, kwargs = _extract_call_input(
                    input_value,
                    kwargs,
                    input_parameter.name,
                )
                native_input = (
                    _wrap_dynamic(input_model, input_value)
                    if input_is_dynamic
                    else _require_model(input_value, input_model)
                )
                contract_start = monotonic()
                try:
                    result, processed_input, stage = self._contract._execute(
                        self._engine,
                        native_input,
                        remedy=self._remedy,
                    )
                except Exception as error:
                    result = ContractResult[object](
                        value=None,
                        succeeded=False,
                        attempts=0,
                        errors=(str(error) or type(error).__name__,),
                    )
                    processed_input = native_input
                    exception: Exception | None = error
                else:
                    exception = _violation(stage, result)

                contract_result = _unwrap(result.value) if result.value is not None else None
                self.contract_successful = result.succeeded
                self.contract_result = contract_result
                self.contract_exception = exception
                _set_legacy_state(
                    self._inner,
                    result.succeeded,
                    contract_result,
                    exception,
                )

                forward_input = processed_input if result.succeeded else native_input
                raw_forward_input = _unwrap(forward_input) if input_is_dynamic else forward_input
                forward_kwargs = dict(kwargs)
                forward_kwargs.pop("validation_context", None)
                forward_start = monotonic()
                try:
                    if input_parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                        forward_kwargs[input_parameter.name] = raw_forward_input
                        forward_result = original_forward(
                            self._inner,
                            *args,
                            **forward_kwargs,
                        )
                    else:
                        forward_result = original_forward(
                            self._inner,
                            raw_forward_input,
                            *args,
                            **forward_kwargs,
                        )
                finally:
                    forward_duration = monotonic() - forward_start
                    self._timings.append(
                        _Timing(
                            contract_s=monotonic() - contract_start,
                            forward_s=forward_duration,
                            attempts=result.attempts,
                        )
                    )

                if options.graceful:
                    return forward_result
                try:
                    return output_adapter.validate_python(forward_result, strict=True)
                except ValidationError as error:
                    msg = (
                        "Legacy contract forward returned a value incompatible with "
                        f"{output_annotation!r}"
                    )
                    raise TypeError(msg) from error

            def contract_perf_stats(self) -> dict[str, dict[str, float | int]]:
                """Return legacy-compatible aggregate execution timings."""
                return _performance_stats(self._timings)

            def __getattr__(self, name: str) -> object:
                return getattr(self._inner, name)

        LegacyContract.__name__ = legacy_type.__name__
        LegacyContract.__qualname__ = legacy_type.__qualname__
        LegacyContract.__module__ = legacy_type.__module__
        LegacyContract.__doc__ = legacy_type.__doc__
        return LegacyContract

    return decorate


def _decorator_options(
    *,
    pre_remedy: bool,
    post_remedy: bool,
    accumulate_errors: bool,
    graceful: bool | None,
    remedy_retry_params: Mapping[str, object] | RetryParams | None,
    verbose: bool,
) -> _DecoratorOptions:
    if isinstance(remedy_retry_params, RetryParams):
        retry = remedy_retry_params
        legacy_graceful = False
    else:
        raw = dict(remedy_retry_params or {})
        legacy_graceful = bool(raw.pop("graceful", False))
        raw.pop("dynamic_engine", None)
        retry = RetryParams.model_validate(raw)
    return _DecoratorOptions(
        pre_remedy=pre_remedy,
        post_remedy=post_remedy,
        accumulate_errors=accumulate_errors,
        graceful=legacy_graceful if graceful is None else graceful,
        retry=retry,
        verbose=verbose,
    )


def _extract_call_input(
    provided: object,
    kwargs: dict[str, object],
    parameter_name: str,
) -> tuple[object, dict[str, object]]:
    names = (parameter_name,) if parameter_name == "input" else (parameter_name, "input")
    present = [name for name in names if name in kwargs]
    if provided is not _MISSING:
        if present:
            msg = "Contract input was provided more than once"
            raise TypeError(msg)
        return provided, kwargs
    if len(present) != 1:
        msg = "Contract call requires exactly one input"
        raise TypeError(msg)
    return kwargs.pop(present[0]), kwargs


def _input_parameter(signature: inspect.Signature) -> inspect.Parameter:
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            if parameter.annotation is inspect.Parameter.empty:
                msg = "Contract forward input requires a type annotation"
                raise TypeError(msg)
            return parameter
    msg = "Contract forward requires one typed input parameter"
    raise TypeError(msg)


def _model_for_annotation(
    annotation: object,
    context: str,
) -> tuple[type[LLMDataModel], bool]:
    if annotation is inspect.Signature.empty:
        msg = f"Contract {context} requires a type annotation"
        raise TypeError(msg)
    if isinstance(annotation, type) and issubclass(annotation, LLMDataModel):
        return annotation, False
    try:
        return build_dynamic_llm_datamodel(annotation), True
    except Exception as error:
        msg = f"Contract {context} annotation is not supported by Pydantic"
        raise TypeError(msg) from error


def _condition(
    inner: object,
    name: str,
    *,
    unwrap: bool,
) -> Callable[[LLMDataModel], None]:
    method = getattr(inner, name, None)
    if not callable(method):
        return _noop

    def validate(value: LLMDataModel) -> None:
        result = method(_unwrap(value) if unwrap else value)
        if result is False:
            msg = f"{name} condition returned false"
            raise ValueError(msg)

    return validate


def _act(inner: object, unwrap_input: bool) -> Callable[[LLMDataModel], LLMDataModel]:
    method = getattr(inner, "act", None)
    if not callable(method):
        return _identity
    signature = inspect.signature(method)
    output_model, output_is_dynamic = _model_for_annotation(
        signature.return_annotation,
        "act return",
    )

    def apply(value: LLMDataModel) -> LLMDataModel:
        result = method(_unwrap(value) if unwrap_input else value)
        if output_is_dynamic:
            return _wrap_dynamic(output_model, result)
        return _require_model(result, output_model)

    return apply


def _require_model(value: object, model: type[LLMDataModel]) -> LLMDataModel:
    if not isinstance(value, model):
        msg = f"Contract input must be {model.__name__}"
        raise TypeError(msg)
    return value


def _wrap_dynamic(model: type[LLMDataModel], value: object) -> LLMDataModel:
    return model.model_validate({"value": value})


def _unwrap(value: object) -> object:
    return getattr(value, "value", value)


def _set_legacy_state(
    inner: object,
    succeeded: bool,
    result: object | None,
    exception: Exception | None,
) -> None:
    state = cast("_LegacyState", inner)
    state.contract_successful = succeeded
    state.contract_result = result
    state.contract_exception = exception


def _violation(
    stage: ContractStage | None,
    result: ContractResult[object],
) -> ContractViolation | None:
    if result.succeeded:
        return None
    if stage is None:
        msg = "Failed native contract did not report a validation stage"
        return ContractViolation("type", (*result.errors, msg))
    return ContractViolation(stage, result.errors)


def _performance_stats(timings: list[_Timing]) -> dict[str, dict[str, float | int]]:
    contract_times = [timing.contract_s for timing in timings]
    forward_times = [timing.forward_s for timing in timings]
    return {
        "contract_execution": _summarize(contract_times),
        "forward_execution": _summarize(forward_times),
        "attempts": {
            "count": len(timings),
            "total": sum(timing.attempts for timing in timings),
        },
    }


def _summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total": sum(values),
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _noop(_value: LLMDataModel) -> None:
    pass


def _identity(value: LLMDataModel) -> LLMDataModel:
    return value
