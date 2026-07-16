from collections.abc import Sequence

import pytest
from pydantic import Field, ValidationError

from symai.contract.models import LLMDataModel
from symai.contract.validation import (
    build_remedy_prompt,
    check_semantic_conditions,
    parse_output,
    validation_errors,
)
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


class Verdict(LLMDataModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class RecordingEngine:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = iter(responses)
        self.requests: list[LanguageModelRequest] = []

    def execute(self, request: LanguageModelRequest, /) -> LanguageModelResponse:
        self.requests.append(request)
        return language_response(next(self.responses))

    def close(self) -> None:
        pass


def language_response(text: str) -> LanguageModelResponse:
    return LanguageModelResponse(
        outputs=(
            LanguageModelOutput(
                index=0,
                message=AssistantOutputMessage(content=(TextContent(text=text),)),
                finish_reason=FinishReason.STOP,
            ),
        ),
        metadata=ResponseMetadata(provider="test", requested_model="judge", status_code=200),
    )


def test_parse_output_returns_typed_model_and_reports_pydantic_paths() -> None:
    assert parse_output('{"label":"positive","confidence":0.8}', Verdict) == Verdict(
        label="positive",
        confidence=0.8,
    )

    with pytest.raises(ValidationError) as caught:
        parse_output('{"label":"positive","confidence":2}', Verdict)

    assert any("confidence" in error for error in validation_errors(caught.value))


def test_remedy_prompt_contains_contract_context_and_selected_errors() -> None:
    prompt = build_remedy_prompt(
        instruction="Classify sentiment.",
        input_text="text: useful",
        output_type=Verdict,
        output_text='{"label":"positive","confidence":2}',
        errors=("confidence must be at most 1",),
    )

    assert "Classify sentiment." in prompt
    assert "text: useful" in prompt
    assert "confidence must be at most 1" in prompt
    assert "Return only the corrected JSON object" in prompt


def test_semantic_conditions_are_judged_through_the_bound_model() -> None:
    engine = RecordingEngine(
        ('{"valid":false,"errors":["label contradicts the review"]}',),
    )
    runtime = Runtime(language_models={"judge": engine})

    with runtime:
        errors = check_semantic_conditions(
            runtime.language_model("judge"),
            Verdict(label="positive", confidence=0.8),
            ("The label must agree with the review.",),
        )

    assert errors == ("label contradicts the review",)
    request = engine.requests[0]
    assert isinstance(request.response_format, JsonSchemaResponseFormat)
    assert "The label must agree with the review." in request.messages[0].content[0].text
