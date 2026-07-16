from __future__ import annotations

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
    def __init__(self, responses: tuple[str, ...]) -> None:
        self._responses = iter(responses)

    def execute(self, _request: LanguageModelRequest, /) -> LanguageModelResponse:
        return LanguageModelResponse(
            outputs=(
                LanguageModelOutput(
                    index=0,
                    message=AssistantOutputMessage(
                        content=(TextContent(text=next(self._responses)),)
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


def test_decorator_resolves_postponed_forward_and_act_annotations() -> None:
    @contract()
    class Normalize:
        class NormalizedReview(LLMDataModel):
            text: str

        prompt = "Normalize the review."

        def act(self, input_value: Review) -> NormalizedReview:
            return self.NormalizedReview(text=input_value.text.casefold())

        def forward(self, input_value: Review) -> Verdict:
            assert isinstance(input_value, self.NormalizedReview)
            return Verdict(label=input_value.text)

    @contract()
    class Total:
        prompt = "Sum the values."

        def act(self, values: list[int]) -> list[int]:
            return sorted(values)

        def forward(self, values: list[int]) -> int:
            assert values == [1, 2, 3]
            return sum(values)

    engine = RecordingEngine(('{"label":"generated"}', '{"value":6}'))
    runtime = Runtime(language_models={"smart": engine})

    with runtime:
        normalized = Normalize(runtime.language_model("smart"))(Review(text="LOUD"))
        total = Total(runtime.language_model("smart"))([3, 1, 2])

    assert normalized == Verdict(label="loud")
    assert total == 6
