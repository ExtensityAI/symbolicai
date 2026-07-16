from collections.abc import Sequence

import pytest

from symai.contract.contract import ContractViolation
from symai.contract.decorator import contract
from symai.contract.models import LLMDataModel
from symai.runtime.models import (
    AssistantOutputMessage,
    FinishReason,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    ResponseMetadata,
    TextContent,
)
from symai.runtime.runtime import Runtime


class Review(LLMDataModel):
    text: str


class Verdict(LLMDataModel):
    label: str


class RecordingEngine:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = iter(responses)
        self.requests: list[LanguageModelRequest] = []

    def execute(self, request: LanguageModelRequest, /) -> LanguageModelResponse:
        self.requests.append(request)
        return LanguageModelResponse(
            outputs=(
                LanguageModelOutput(
                    index=0,
                    message=AssistantOutputMessage(
                        content=(TextContent(text=next(self.responses)),)
                    ),
                    finish_reason=FinishReason.STOP,
                ),
            ),
            metadata=ResponseMetadata(
                provider="test",
                requested_model="contract",
                status_code=200,
            ),
        )

    def close(self) -> None:
        pass


def test_decorator_runs_native_contract_then_original_forward() -> None:
    @contract()
    class Classify:
        prompt = "Classify sentiment."

        def __init__(self, prefix: str) -> None:
            self.prefix = prefix
            self.forward_calls = 0

        def forward(self, _input: Review) -> Verdict:
            self.forward_calls += 1
            assert self.contract_successful is True
            assert self.contract_exception is None
            return self.contract_result

    engine = RecordingEngine(('{"label":"positive"}',))
    runtime = Runtime(language_models={"smart": engine})

    with runtime:
        classify = Classify(runtime.language_model("smart"), "result")
        verdict = classify(Review(text="Useful"))

    assert verdict == Verdict(label="positive")
    assert classify.prefix == "result"
    assert classify.forward_calls == 1
    assert classify.contract_result == verdict
    assert classify.contract_perf_stats()["contract_execution"]["count"] == 1


def test_decorator_preserves_forward_fallback_after_contract_failure() -> None:
    @contract()
    class Fallback:
        prompt = "Classify sentiment."

        def pre(self, _input: Review) -> None:
            msg = "rejected"
            raise ValueError(msg)

        def forward(self, input_value: Review) -> Verdict:
            assert self.contract_successful is False
            assert isinstance(self.contract_exception, ContractViolation)
            return Verdict(label=f"fallback:{input_value.text}")

    engine = RecordingEngine(())
    runtime = Runtime(language_models={"smart": engine})

    with runtime:
        fallback = Fallback(runtime.language_model("smart"))
        verdict = fallback(input=Review(text="raw"))

    assert verdict == Verdict(label="fallback:raw")
    assert fallback.contract_result is None
    assert engine.requests == []


def test_decorator_wraps_and_unwraps_native_type_annotations() -> None:
    @contract()
    class Upper:
        prompt = "Uppercase the value."

        def forward(self, _input: str) -> str:
            assert self.contract_result == "HELLO"
            return self.contract_result

    engine = RecordingEngine(('{"value":"HELLO"}',))
    runtime = Runtime(language_models={"smart": engine})

    with runtime:
        result = Upper(runtime.language_model("smart"))("hello")

    assert result == "HELLO"


def test_graceful_decorator_mode_allows_an_untyped_fallback_result() -> None:
    @contract(remedy_retry_params={"graceful": True})
    class Graceful:
        prompt = "Return text."

        def pre(self, _input: Review) -> None:
            msg = "rejected"
            raise ValueError(msg)

        def forward(self, _input: Review) -> str:
            return 246  # type: ignore[return-value]

    engine = RecordingEngine(())
    runtime = Runtime(language_models={"smart": engine})

    with runtime:
        result = Graceful(runtime.language_model("smart"))(Review(text="raw"))

    assert result == 246


def test_decorator_validates_the_legacy_class_at_construction() -> None:
    @contract()
    class MissingPrompt:
        def forward(self, input_value: Review) -> Verdict:
            return Verdict(label=input_value.text)

    engine = RecordingEngine(())
    runtime = Runtime(language_models={"smart": engine})

    with runtime, pytest.raises(TypeError, match="prompt"):
        MissingPrompt(runtime.language_model("smart"))


def test_decorator_rejects_unknown_retry_options() -> None:
    with pytest.raises(ValueError, match="unknown"):
        contract(remedy_retry_params={"unknown": 1})
