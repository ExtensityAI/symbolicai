from collections.abc import Callable

import pytest

import symai.ops as ops
from symai.operations import (
    combine_request,
    equals_request,
    include_request,
    map_request,
    query_request,
    setitem_request,
    summarize_request,
)
from symai.ops import SYMBOL_PRIMITIVES, primitives
from symai.runtime.errors import NoActiveRuntimeError
from symai.runtime.models import (
    AssistantOutputMessage,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FinishReason,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    Provider,
    ResponseMetadata,
    SystemMessage,
    TextContent,
    UserMessage,
)
from symai.runtime.runtime import Runtime
from symai.symbol import Expression, Symbol

METADATA = ResponseMetadata(
    provider=Provider.OPENAI,
    requested_model="recording-model",
    status_code=200,
    request_id="request-10",
)


def language_response(*outputs: tuple[int, str]) -> LanguageModelResponse:
    return LanguageModelResponse(
        outputs=tuple(
            LanguageModelOutput(
                index=index,
                message=AssistantOutputMessage(content=(TextContent(text=text),)),
                finish_reason=FinishReason.STOP if text else FinishReason.CONTENT_FILTER,
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

    def close(self) -> None:
        pass


class RecordingEmbeddingEngine:
    def __init__(self, response: EmbeddingResponse) -> None:
        self.requests: list[EmbeddingRequest] = []
        self._response = response

    def execute(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return self._response

    def close(self) -> None:
        pass


def runtime_for(
    *,
    language: RecordingLanguageEngine | None = None,
    embedding: RecordingEmbeddingEngine | None = None,
) -> Runtime:
    language_models = (
        {"test-language-model": language} if language is not None else None
    )
    embeddings = {"test-embedding": embedding} if embedding is not None else None
    return Runtime(
        language_models=language_models,
        embeddings=embeddings,
        default_language_model="test-language-model" if language is not None else None,
        default_embedding="test-embedding" if embedding is not None else None,
    )


def test_semantic_equality_uses_normalized_request_index_and_metadata() -> None:
    engine = RecordingLanguageEngine(
        lambda _request: language_response((0, "false"), (1, "  'yes'  "))
    )

    with runtime_for(language=engine):
        result, metadata = Symbol("one", semantic=True).equals(
            1,
            output_index=1,
            return_metadata=True,
        )

    assert result.value is True
    assert metadata is METADATA
    assert engine.requests == [equals_request("one", 1)]


def test_symbol_method_request_includes_static_and_dynamic_context() -> None:
    engine = RecordingLanguageEngine(lambda _request: language_response((0, "summary")))
    value = Symbol(
        "long",
        semantic=True,
        static_context="Stable facts.",
        dynamic_context="Current request.",
    )

    with runtime_for(language=engine):
        result = value.summarize()

    base = summarize_request("long")
    base_system = base.messages[0]
    assert isinstance(base_system, SystemMessage)
    assert result.value == "summary"
    assert engine.requests == [
        base.model_copy(
            update={
                "messages": (
                    SystemMessage(
                        content=(
                            TextContent(
                                text=(
                                    f"{base_system.content[0].text}\n"
                                    "<STATIC_CONTEXT/>\nStable facts.\n"
                                    "<DYNAMIC_CONTEXT/>\nCurrent request."
                                )
                            ),
                        )
                    ),
                    *base.messages[1:],
                )
            }
        )
    ]


def test_semantic_dunder_returns_native_success_without_a_runtime() -> None:
    result = Symbol(1, semantic=True) + 2

    assert result.value == 3


def test_semantic_dunder_falls_back_only_for_unsupported_native_operation() -> None:
    engine = RecordingLanguageEngine(lambda _request: language_response((0, "combined")))

    with runtime_for(language=engine):
        result = Symbol(["one"], semantic=True) + "two"

    assert result.value == "combined"
    assert engine.requests == [combine_request(["one"], "two")]


def test_semantic_dunder_propagates_unrelated_native_exception() -> None:
    class ExplodingValue:
        def __add__(self, _other: object) -> object:
            msg = "native bug"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="native bug"):
        _ = Symbol(ExplodingValue(), semantic=True) + 1


def test_semantic_dunder_propagates_native_type_error() -> None:
    class ExplodingValue:
        def __add__(self, _other: object) -> object:
            msg = "native type bug"
            raise TypeError(msg)

    engine = RecordingLanguageEngine(lambda _request: language_response((0, "fallback")))
    with runtime_for(language=engine), pytest.raises(TypeError, match="native type bug"):
        _ = Symbol(ExplodingValue(), semantic=True) + 1

    assert engine.requests == []


def test_engine_backed_symbol_path_requires_active_runtime() -> None:
    with pytest.raises(NoActiveRuntimeError):
        Symbol("long text", semantic=True).summarize()


def test_native_symbol_paths_remain_runtime_free() -> None:
    value = Symbol(["first", "second"])

    assert value[1] == "second"  # pyright: ignore[reportIndexIssue]
    assert Symbol("alphabet").startswith("alpha") is True
    value[0] = "changed"  # pyright: ignore[reportIndexIssue]
    assert value.value == ["changed", "second"]


def test_scalar_iteration_remains_runtime_free() -> None:
    assert list(Symbol("single")) == ["single"]


def test_semantic_item_mutation_parses_literal_and_mutates_in_place() -> None:
    responses = iter(
        (
            language_response((0, "ignored"), (2, "  '[1, 9, 3]'  ")),
            language_response((0, "[1, 3]")),
        )
    )
    engine = RecordingLanguageEngine(lambda _request: next(responses))
    value = Symbol([1, 2, 3], semantic=True)

    with runtime_for(language=engine):
        value.__setitem__(1, 9, output_index=2)
        del value[1]  # pyright: ignore[reportIndexIssue]

    assert value.value == [1, 3]
    assert engine.requests == [
        setitem_request(Symbol([1, 2, 3], semantic=True), 1, 9),
        setitem_request(Symbol([1, 9, 3], semantic=True), 1, None, delete=True),
    ]


def test_representative_language_methods_preserve_requests_and_typed_results() -> None:
    responses = iter(
        (
            language_response((0, "  'answer'  ")),
            language_response((0, "['ONE', 'TWO', 'THREE']")),
            language_response((0, "included")),
        )
    )
    engine = RecordingLanguageEngine(lambda _request: next(responses))

    with runtime_for(language=engine):
        queried = Symbol("facts", semantic=True).query("question")
        mapped = Symbol(["one", "two", "three"], semantic=True).map(
            "uppercase",
            limit=2,
        )
        included = Symbol("base", semantic=True).include("detail")

    assert queried.value == "answer"
    assert mapped.value == ["ONE", "TWO"]
    assert included.value == "included"
    assert engine.requests == [
        query_request("facts", "question"),
        map_request(["one", "two", "three"], "uppercase"),
        include_request("base", "detail"),
    ]


def test_summary_default_is_explicit_and_limited_after_fallback() -> None:
    engine = RecordingLanguageEngine(lambda _request: language_response((0, "unused"), (3, "")))

    with runtime_for(language=engine):
        result = Symbol("long", semantic=True).summarize(
            output_index=3,
            default=["fallback", "extra"],
            return_type=list,
            limit=1,
        )

    assert result.value == ["fallback"]
    assert engine.requests == [summarize_request("long")]


def test_embedding_uses_normalized_request_and_provider_index_order() -> None:
    engine = RecordingEmbeddingEngine(
        EmbeddingResponse(
            vectors=(
                EmbeddingVector(index=1, values=(3.0, 4.0)),
                EmbeddingVector(index=0, values=(1.0, 2.0)),
            ),
            metadata=METADATA,
        )
    )

    with runtime_for(embedding=engine):
        result, metadata = Symbol(
            ["first", "second"],
            static_context="Do not send to embedding.",
            dynamic_context="Also do not send.",
        ).embed(
            dimensions=2,
            user="caller",
            return_metadata=True,
        )

    assert result.value == [[1.0, 2.0], [3.0, 4.0]]
    assert all(type(value) is float for vector in result.value for value in vector)
    assert metadata is METADATA
    assert engine.requests == [
        EmbeddingRequest(inputs=("first", "second"), dimensions=2, user="caller")
    ]


@pytest.mark.parametrize("value", [1, b"bytes", bytearray(b"bytes")])
def test_embedding_rejects_non_text_inputs(value: object) -> None:
    with pytest.raises(TypeError, match="text"):
        Symbol(value).embed()


def test_expression_prompt_uses_explicit_runtime() -> None:
    engine = RecordingLanguageEngine(
        lambda _request: language_response((1, "second"), (0, "first"))
    )

    with runtime_for(language=engine):
        result = Expression.prompt("Say hello", output_index=1)

    assert isinstance(result, Expression)
    assert result.value == "second"
    assert engine.requests == [
        LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="Say hello"),)),))
    ]


def test_ops_namespace_is_exact_and_explicit() -> None:
    assert {name for name in vars(ops) if not name.startswith("_")} == {
        "SYMBOL_PRIMITIVES",
        "primitives",
    }


def test_unsupported_public_capabilities_are_absent() -> None:
    for name in (
        "command",
        "open",
        "config",
        "add",
        "get",
        "tune",
        "cluster",
        "simulate",
        "analyze",
        "stream",
        "tokenizer",
        "tokens",
    ):
        assert not hasattr(Symbol, name)
    deleted_classes = (
        "DataClusteringPrimitives",
        "DictHandlingPrimitives",
        "ExecutionControlPrimitives",
        "FineTuningPrimitives",
        "IOHandlingPrimitives",
        "IndexingPrimitives",
        "UniquenessPrimitives",
    )
    assert all(not hasattr(primitives, name) for name in deleted_classes)
    assert all(type_.__name__ not in deleted_classes for type_ in SYMBOL_PRIMITIVES)
    for name in (
        "Call",
        "GlobalSymbolPrimitive",
        "Metadata",
        "calculate_frechet_distance",
        "primitive",
        "_global_primitive",
    ):
        assert not hasattr(Symbol, name)
        assert not hasattr(primitives, name)
    value = Symbol([1.0])
    for name in ("calculate_frechet_distance", "primitive", "_global_primitive"):
        assert not hasattr(value, name)
