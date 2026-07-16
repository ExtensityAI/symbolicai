import inspect
from collections.abc import Callable, Sequence
from importlib.util import find_spec
from typing import get_type_hints

import pytest

from symai.function import Function
from symai.runtime.models import (
    AssistantOutputMessage,
    FinishReason,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    ResponseMetadata,
    SamplingConfig,
    SystemMessage,
    TextContent,
    TokenUsage,
    UserMessage,
)
from symai.runtime.runtime import Runtime

METADATA = ResponseMetadata(
    provider="openai",
    requested_model="test-model",
    status_code=200,
    request_id="request-1",
    usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
)


def response(text: str) -> LanguageModelResponse:
    return LanguageModelResponse(
        outputs=(
            LanguageModelOutput(
                index=0,
                message=AssistantOutputMessage(content=(TextContent(text=text),)),
                finish_reason=FinishReason.STOP,
            ),
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

    def close(self) -> None:
        pass


def runtime_for(engines: dict[str, RecordingLanguageEngine]) -> Runtime:
    return Runtime(language_models=engines)


def user_text(request: LanguageModelRequest) -> str:
    message = request.messages[-1]
    assert isinstance(message, UserMessage)
    return "".join(part.text for part in message.content if isinstance(part, TextContent))


def test_request_builds_normalized_request_without_execution() -> None:
    engine = RecordingLanguageEngine(lambda _request: response("unused"))
    function = Function(
        "Answer precisely.",
        examples=("2 + 2 => 4",),
        max_tokens=64,
        stop=("END",),
    )

    request = function.request("3 +", 4)

    assert request == LanguageModelRequest(
        messages=(
            SystemMessage(content=(TextContent(text="Answer precisely.\n2 + 2 => 4"),)),
            UserMessage(content=(TextContent(text="3 + 4"),)),
        ),
        sampling=SamplingConfig(max_tokens=64, stop=("END",)),
    )
    assert engine.requests == []


def test_examples_accept_only_strings_and_snapshot_sequences() -> None:
    mutable_examples = ["first"]
    function = Function(examples=mutable_examples)
    mutable_examples.append("second")

    assert function.examples == ("first",)
    assert Function(examples="only").examples == ("only",)
    with pytest.raises(TypeError, match="examples must contain only strings"):
        Function(examples=("valid", 1))  # pyright: ignore[reportArgumentType]


def test_examples_annotation_has_no_prompt_hierarchy() -> None:
    annotation = get_type_hints(Function.__init__)["examples"]

    assert annotation == Sequence[str] | str | None


def test_call_returns_exact_normalized_response_and_forwards_engine() -> None:
    unused = RecordingLanguageEngine(lambda _request: response("unused"))
    expected = response("selected")
    selected = RecordingLanguageEngine(lambda _request: expected)
    function = Function("Answer.")

    with runtime_for({"unused": unused, "tenant-a": selected}) as runtime:
        actual = function(runtime.language_model("tenant-a"), "question")

    assert actual is expected
    assert actual.metadata is METADATA
    assert unused.requests == []
    assert tuple(user_text(request) for request in selected.requests) == ("question",)


def test_execute_many_is_sequential_and_preserves_nested_input_order() -> None:
    events: list[str] = []
    returned: list[LanguageModelResponse] = []

    def execute(request: LanguageModelRequest) -> LanguageModelResponse:
        text = user_text(request)
        events.append(text)
        result = response(text)
        returned.append(result)
        return result

    engine = RecordingLanguageEngine(execute)
    function = Function("Echo.")

    with runtime_for({"ordered": engine}) as runtime:
        results = function.execute_many(
            runtime.language_model(),
            (("first", 1), ("second", 2), ("third", 3)),
        )

    assert events == ["first 1", "second 2", "third 3"]
    assert results == tuple(returned)
    assert all(actual is expected for actual, expected in zip(results, returned, strict=True))


def test_function_has_one_non_generic_execution_surface() -> None:
    init_parameters = inspect.signature(Function).parameters
    call_parameters = inspect.signature(Function.__call__).parameters

    assert not {
        "default",
        "dynamic_context",
        "limit",
        "return_type",
        "static_context",
        "sym_return_type",
    }.intersection(init_parameters)
    assert not {"engine", "preview", "return_metadata", "output_index"}.intersection(
        call_parameters
    )
    assert not hasattr(Function, "batch")
    assert get_type_hints(Function.request)["return"] is LanguageModelRequest
    assert get_type_hints(Function.__call__)["return"] is LanguageModelResponse


def test_legacy_components_module_is_deleted() -> None:
    assert find_spec("symai.components") is None
