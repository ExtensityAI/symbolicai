from collections.abc import Sequence
from typing import Any, override

import pytest
from pydantic import Field

from symai.contract.contract import ContractViolation
from symai.contract.decorator import contract
from symai.contract.models import LLMDataModel
from symai.runtime.errors import ErrorMetadata, TransportError
from symai.runtime.models import (
    AssistantOutputMessage,
    FinishReason,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    ResponseMetadata,
    TextContent,
    TokenUsage,
)
from symai.runtime.runtime import Runtime


class Review(LLMDataModel):
    text: str = Field(description="Raw review text")


class Verdict(LLMDataModel):
    label: str


class LegacyState:
    contract_successful: bool
    contract_result: Any
    contract_exception: Exception | None


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
    class Classify(LegacyState):
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
    class Fallback(LegacyState):
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
    class Upper(LegacyState):
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


def test_decorator_passes_post_act_input_to_generation_and_failure_fallback() -> None:
    @contract(post_remedy=False)
    class Normalize(LegacyState):
        prompt = "Classify sentiment."

        def act(self, input_value: Review) -> Review:
            return Review(text=input_value.text.casefold())

        def forward(self, input_value: Review) -> Verdict:
            assert self.contract_successful is False
            return Verdict(label=f"fallback:{input_value.text}")

    engine = RecordingEngine(('{"not_label":"invalid"}',))
    runtime = Runtime(language_models={"smart": engine})

    with runtime:
        verdict = Normalize(runtime.language_model("smart"))(Review(text="LOUD"))

    assert verdict == Verdict(label="fallback:loud")
    content = engine.requests[0].messages[-1].content[0]
    assert isinstance(content, TextContent)
    assert content.text == "text: loud\n  Description: Raw review text"


def test_decorator_rejects_an_act_result_that_is_not_its_annotated_model() -> None:
    @contract()
    class InvalidAct:
        prompt = "Classify sentiment."

        def act(self, _input: Review) -> Review:
            return object()  # pyright: ignore[reportReturnType]

        def forward(self, _input: Review) -> Verdict:
            pytest.fail("forward must not mask an act programming error")

    engine = RecordingEngine(())
    runtime = Runtime(language_models={"smart": engine})

    with runtime, pytest.raises(TypeError, match="Review"):
        InvalidAct(runtime.language_model("smart"))(Review(text="input"))

    assert engine.requests == []


def test_decorator_propagates_transport_errors_without_running_fallback() -> None:
    failure = TransportError(
        "transport failed",
        metadata=ErrorMetadata(provider="test", model="contract"),
    )

    class FailingEngine(RecordingEngine):
        @override
        def execute(self, request: LanguageModelRequest, /) -> LanguageModelResponse:
            self.requests.append(request)
            raise failure

    @contract()
    class NoFallback:
        prompt = "Classify sentiment."

        def __init__(self) -> None:
            self.forward_calls = 0

        def forward(self, _input: Review) -> Verdict:
            self.forward_calls += 1
            return Verdict(label="fallback")

    engine = FailingEngine(())
    runtime = Runtime(language_models={"smart": engine})

    with runtime:
        classify = NoFallback(runtime.language_model("smart"))
        with pytest.raises(TransportError) as caught:
            classify(Review(text="input"))

    assert caught.value is failure
    assert classify.forward_calls == 0
    assert classify.contract_result is None


def test_contract_perf_stats_account_for_every_observed_model_execution() -> None:
    class AccountingEngine(RecordingEngine):
        def __init__(self, responses: Sequence[str], provider: str) -> None:
            super().__init__(responses)
            self.provider = provider

        @override
        def execute(self, request: LanguageModelRequest, /) -> LanguageModelResponse:
            response = super().execute(request)
            metadata = response.metadata.model_copy(
                update={
                    "provider": self.provider,
                    "usage": TokenUsage(
                        prompt_tokens=1,
                        completion_tokens=2,
                        total_tokens=3,
                    ),
                }
            )
            return response.model_copy(update={"metadata": metadata})

    @contract(post_remedy=True, remedy_retry_params={"delay": 0})
    class SemanticallyChecked(LegacyState):
        prompt = "Classify sentiment."
        semantic_conditions = ("The label must be approved.",)

        def forward(self, _input: Review) -> Verdict:
            return self.contract_result

    primary_engine = AccountingEngine(('{"label":"draft"}',), "primary")
    remedy_engine = AccountingEngine(
        (
            '{"valid":false,"errors":["label is not approved"]}',
            '{"label":"approved"}',
            '{"valid":true,"errors":[]}',
        ),
        "remedy",
    )
    primary_runtime = Runtime(language_models={"primary": primary_engine})
    remedy_runtime = Runtime(language_models={"remedy": remedy_engine})

    with primary_runtime, remedy_runtime:
        checked = SemanticallyChecked(
            primary_runtime.language_model("primary"),
            remedy=remedy_runtime.language_model("remedy"),
        )
        result = checked(Review(text="input"))

    stats = checked.contract_perf_stats()
    assert result == Verdict(label="approved")
    assert stats["contract_execution"]["count"] == 4
    assert stats["usage"]["prompt_tokens"] == 4
    assert stats["usage"]["completion_tokens"] == 8
    assert stats["providers"]["primary"]["count"] == 1
    assert stats["providers"]["remedy"]["count"] == 3
    assert len(stats["executions"]) == 4
