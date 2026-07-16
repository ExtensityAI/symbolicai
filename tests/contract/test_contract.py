from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest
from pydantic import Field, ValidationError

from symai.contract.contract import (
    Contract,
    ContractOptions,
    ContractViolation,
    RetryParams,
)
from symai.contract.models import LLMDataModel
from symai.runtime.models import (
    AssistantOutputMessage,
    FinishReason,
    JsonSchemaResponseFormat,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    ResponseMetadata,
    TextContent,
)
from symai.runtime.runtime import Runtime

if TYPE_CHECKING:
    from collections.abc import Sequence

    from symai.runtime.observability import ExecutionRecord


class Review(LLMDataModel):
    text: str


class Verdict(LLMDataModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class Analysis(LLMDataModel):
    normalized_text: str


class RecordingEngine:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = iter(responses)
        self.requests: list[LanguageModelRequest] = []

    def execute(self, request: LanguageModelRequest, /) -> LanguageModelResponse:
        self.requests.append(request)
        text = next(self.responses)
        return LanguageModelResponse(
            outputs=(
                LanguageModelOutput(
                    index=0,
                    message=AssistantOutputMessage(content=(TextContent(text=text),)),
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


def test_native_contract_generates_and_returns_typed_output() -> None:
    engine = RecordingEngine(('{"label":"positive","confidence":0.9}',))
    runtime = Runtime(language_models={"smart": engine})
    classify = Contract(
        instruction="Classify sentiment.",
        input_type=Review,
        output_type=Verdict,
    )

    with runtime:
        verdict = classify(runtime.language_model("smart"), Review(text="Useful"))

    assert verdict == Verdict(label="positive", confidence=0.9)
    assert isinstance(engine.requests[0].response_format, JsonSchemaResponseFormat)
    assert "text: Useful" in engine.requests[0].messages[-1].content[0].text


def test_post_failure_is_remedied_with_a_bounded_second_generation() -> None:
    records: list[ExecutionRecord] = []
    engine = RecordingEngine(
        (
            '{"label":"positive","confidence":0.2}',
            '{"label":"positive","confidence":0.9}',
        )
    )
    runtime = Runtime(language_models={"smart": engine}, observers=(records.append,))

    def require_confidence(output: Verdict) -> None:
        if output.confidence < 0.8:
            msg = "confidence must be at least 0.8"
            raise ValueError(msg)

    classify = Contract(
        instruction="Classify sentiment.",
        input_type=Review,
        output_type=Verdict,
        post=require_confidence,
        options=ContractOptions(retry=RetryParams(tries=1, delay=0)),
    )

    with runtime:
        result = classify.run(runtime.language_model("smart"), Review(text="Useful"))

    assert result.succeeded is True
    assert result.value == Verdict(label="positive", confidence=0.9)
    assert result.attempts == 2
    assert result.errors == ("confidence must be at least 0.8",)
    assert len(engine.requests) == 2
    assert [record.engine for record in records] == ["smart", "smart"]


def test_act_can_transform_input_into_a_distinct_prompt_model() -> None:
    engine = RecordingEngine(('{"label":"positive","confidence":0.9}',))
    runtime = Runtime(language_models={"smart": engine})
    classify = Contract(
        instruction="Classify sentiment.",
        input_type=Review,
        output_type=Verdict,
        act=lambda review: Analysis(normalized_text=review.text.casefold()),
    )

    with runtime:
        classify(runtime.language_model("smart"), Review(text="USEFUL"))

    assert "normalized_text: useful" in engine.requests[0].messages[-1].content[0].text


def test_run_reports_exhaustion_and_call_raises_the_stage_failure() -> None:
    engine = RecordingEngine(
        (
            '{"label":"positive","confidence":2}',
            '{"label":"positive","confidence":2}',
            '{"label":"positive","confidence":2}',
            '{"label":"positive","confidence":2}',
        )
    )
    runtime = Runtime(language_models={"smart": engine})
    classify = Contract(
        instruction="Classify sentiment.",
        input_type=Review,
        output_type=Verdict,
        options=ContractOptions(retry=RetryParams(tries=1, delay=0)),
    )

    with runtime:
        result = classify.run(runtime.language_model("smart"), Review(text="Useful"))
        with pytest.raises(ContractViolation) as caught:
            classify(runtime.language_model("smart"), Review(text="Useful"))

    assert result.succeeded is False
    assert result.value is None
    assert result.attempts == 2
    assert caught.value.stage == "type"
    assert any("confidence" in error for error in caught.value.errors)


def test_precondition_failure_without_remedy_performs_no_io() -> None:
    engine = RecordingEngine(())
    runtime = Runtime(language_models={"smart": engine})

    def reject(_input: Review) -> None:
        msg = "review is not eligible"
        raise ValueError(msg)

    classify = Contract(
        instruction="Classify sentiment.",
        input_type=Review,
        output_type=Verdict,
        pre=reject,
    )

    with runtime:
        result = classify.run(runtime.language_model("smart"), Review(text="Useful"))

    assert result.succeeded is False
    assert result.attempts == 0
    assert result.errors == ("review is not eligible",)
    assert engine.requests == []


def test_precondition_can_remedy_input_before_act_and_generation() -> None:
    primary = RecordingEngine(('{"label":"positive","confidence":0.9}',))
    remedy = RecordingEngine(('{"text":"fixed"}',))
    runtime = Runtime(language_models={"primary": primary, "remedy": remedy})

    def require_fixed(input_value: Review) -> None:
        if input_value.text != "fixed":
            msg = "text must be fixed"
            raise ValueError(msg)

    classify = Contract(
        instruction="Classify sentiment.",
        input_type=Review,
        output_type=Verdict,
        pre=require_fixed,
        options=ContractOptions(
            pre_remedy=True,
            retry=RetryParams(tries=1, delay=0),
        ),
    )

    with runtime:
        result = classify.run(
            runtime.language_model("primary"),
            Review(text="bad"),
            remedy=runtime.language_model("remedy"),
        )

    assert result.value == Verdict(label="positive", confidence=0.9)
    assert result.attempts == 2
    assert result.errors == ("text must be fixed",)
    assert "text: fixed" in primary.requests[0].messages[-1].content[0].text


def test_semantic_validation_and_correction_use_the_remedy_handle() -> None:
    primary = RecordingEngine(('{"label":"positive","confidence":0.9}',))
    remedy = RecordingEngine(
        (
            '{"valid":false,"errors":["label contradicts review"]}',
            '{"label":"negative","confidence":0.9}',
            '{"valid":true,"errors":[]}',
        )
    )
    runtime = Runtime(language_models={"primary": primary, "remedy": remedy})
    classify = Contract(
        instruction="Classify sentiment.",
        input_type=Review,
        output_type=Verdict,
        semantic_conditions=("The label must agree with the review.",),
        options=ContractOptions(retry=RetryParams(tries=1, delay=0)),
    )

    with runtime:
        result = classify.run(
            runtime.language_model("primary"),
            Review(text="Terrible"),
            remedy=runtime.language_model("remedy"),
        )

    assert result.value == Verdict(label="negative", confidence=0.9)
    assert result.attempts == 2
    assert len(primary.requests) == 1
    assert len(remedy.requests) == 3


def test_native_contract_configuration_and_results_are_immutable() -> None:
    classify = Contract(
        instruction="Classify sentiment.",
        input_type=Review,
        output_type=Verdict,
    )
    with pytest.raises(FrozenInstanceError):
        classify.instruction = "Changed"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        RetryParams(tries=0)
