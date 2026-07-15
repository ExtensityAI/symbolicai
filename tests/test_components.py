from collections.abc import Callable
from typing import get_type_hints

import pytest

from symai.backend.engine_handle import EngineHandle
from symai.components import Function
from symai.runtime.errors import NoActiveRuntimeError
from symai.runtime.models import (
    AssistantOutputMessage,
    FinishReason,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    Provider,
    ResponseMetadata,
    SamplingConfig,
    SystemMessage,
    TextContent,
    TokenUsage,
    UserMessage,
)
from symai.runtime.runtime import Runtime
from symai.symbol import Symbol

METADATA = ResponseMetadata(
    provider=Provider.OPENAI,
    requested_model="test-model",
    status_code=200,
    request_id="request-1",
    usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
)


def response(*outputs: tuple[int, str]) -> LanguageModelResponse:
    return LanguageModelResponse(
        outputs=tuple(
            LanguageModelOutput(
                index=index,
                message=AssistantOutputMessage(content=(TextContent(text=text),)),
                finish_reason=FinishReason.STOP,
            )
            for index, text in outputs
        ),
        metadata=METADATA,
    )


class RecordingLanguageEngine:
    def __init__(
        self,
        execute: Callable[[LanguageModelRequest], LanguageModelResponse],
    ) -> None:
        self.requests: list[LanguageModelRequest] = []
        self._execute = execute

    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse:
        self.requests.append(request)
        return self._execute(request)


def runtime_for(engine: RecordingLanguageEngine) -> Runtime:
    return Runtime(language_model=EngineHandle(engine, lambda: None))


def user_text(request: LanguageModelRequest) -> str:
    message = request.messages[-1]
    assert isinstance(message, UserMessage)
    return "".join(part.text for part in message.content if isinstance(part, TextContent))


def test_function_builds_normalized_request_and_returns_typed_symbol_with_metadata() -> None:
    engine = RecordingLanguageEngine(lambda _request: response((0, "7")))
    function = Function(
        "Answer with one integer.",
        examples=("2 + 2 => 4",),
        return_type=int,
        static_context="Use arithmetic.",
        dynamic_context="The caller needs precision.",
        max_tokens=64,
        stop=("END",),
    )

    with runtime_for(engine):
        result, metadata = function("3 + 4", return_metadata=True)

    assert isinstance(result, Symbol)
    assert result.value == 7
    assert metadata is METADATA
    assert metadata.usage == TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    assert engine.requests == [
        LanguageModelRequest(
            messages=(
                SystemMessage(
                    content=(
                        TextContent(
                            text=(
                                "Answer with one integer.\n"
                                "<STATIC_CONTEXT/>\nUse arithmetic.\n"
                                "<DYNAMIC_CONTEXT/>\nThe caller needs precision.\n"
                                "2 + 2 => 4"
                            )
                        ),
                    )
                ),
                UserMessage(content=(TextContent(text="3 + 4"),)),
            ),
            sampling=SamplingConfig(max_tokens=64, stop=("END",)),
        )
    ]


def test_function_requires_an_active_runtime_for_execution() -> None:
    with pytest.raises(NoActiveRuntimeError):
        Function("Answer.")("question")


def test_function_applies_explicit_default_and_collection_limit() -> None:
    responses = iter((response((0, "invalid")), response((0, "[1, 2, 3]"))))
    engine = RecordingLanguageEngine(lambda _request: next(responses))

    with runtime_for(engine):
        defaulted = Function("Integer.", return_type=int, default=9)("value")
        limited = Function("List.", return_type=list, limit=2)("values")

    assert defaulted.value == 9
    assert limited.value == [1, 2]


def test_function_rejects_a_default_outside_its_declared_return_type() -> None:
    with pytest.raises(TypeError, match="default"):
        Function("Integer.", return_type=int, default="9")


def test_function_without_default_propagates_typed_parse_failure() -> None:
    engine = RecordingLanguageEngine(lambda _request: response((0, "invalid")))

    with runtime_for(engine), pytest.raises(ValueError, match="invalid literal"):
        Function("Integer.", return_type=int)("value")


def test_function_preview_returns_frozen_request_without_execution() -> None:
    engine = RecordingLanguageEngine(lambda _request: response((0, "unused")))
    function = Function('Return {"answer": 1}.', max_tokens=8)

    with runtime_for(engine):
        preview = function("question", preview=True)

    assert preview == LanguageModelRequest(
        messages=(
            SystemMessage(content=(TextContent(text='Return {"answer": 1}.'),)),
            UserMessage(content=(TextContent(text="question"),)),
        ),
        sampling=SamplingConfig(max_tokens=8),
    )
    assert engine.requests == []


def test_function_public_annotations_resolve_at_runtime() -> None:
    assert get_type_hints(Function.__init__)["examples"]
    assert get_type_hints(Function.__call__)["return"]
    assert get_type_hints(Function.batch)["inputs"]


def test_function_selects_normalized_output_by_declared_index() -> None:
    engine = RecordingLanguageEngine(lambda _request: response((1, "second"), (0, "first")))

    with runtime_for(engine):
        result = Function("Answer.")("question", output_index=1)

    assert result.value == "second"


def test_function_batch_executes_stably_and_supports_batch_preview() -> None:
    engine = RecordingLanguageEngine(
        lambda request: response((0, str(int(user_text(request)) * 2)))
    )
    function = Function("Double.", return_type=int)

    with runtime_for(engine):
        results = function.batch(("1", "2", "3"))
        previews = function.batch(("4", "5"), preview=True)

    assert tuple(result.value for result in results) == (2, 4, 6)
    assert tuple(user_text(request) for request in engine.requests) == ("1", "2", "3")
    assert tuple(user_text(request) for request in previews) == ("4", "5")
    assert len(engine.requests) == 3


def test_function_batch_rejects_one_string_as_a_sequence() -> None:
    function = Function("Answer.")

    with pytest.raises(TypeError, match="sequence"):
        function.batch("not-a-batch")
