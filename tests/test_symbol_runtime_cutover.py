from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

import symai.ops as ops
import symai.runtime.runtime as runtime_module
import symai.symbol as symbol_module
from symai.decoding import ConstructorDecoder, TextDecoder
from symai.function import Function
from symai.operations import embedding_request, language_request
from symai.ops import compare, embed, primitives, rank, reason, text
from symai.prompts import (
    CombineText,
    ContainsValue,
    ExtractPattern,
    Format,
    FuzzyEquals,
    IncludeText,
    IsInstanceOf,
    LogicExpression,
    MapExpression,
    Modify,
    RankList,
    ReplaceText,
    SimpleSymbolicExpression,
)
from symai.runtime.models import (
    AssistantOutputMessage,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FinishReason,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    ResponseMetadata,
    TextContent,
)
from symai.runtime.runtime import Runtime
from symai.symbol import Symbol

METADATA = ResponseMetadata(
    provider="test-provider",
    requested_model="test-model",
    status_code=200,
    request_id="request-8",
)


def language_response(text: str) -> LanguageModelResponse:
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
    def __init__(self, response: LanguageModelResponse) -> None:
        self.requests: list[LanguageModelRequest] = []
        self.response = response

    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse:
        self.requests.append(request)
        return self.response

    def close(self) -> None:
        pass


class RecordingEmbeddingEngine:
    def __init__(self, response: EmbeddingResponse) -> None:
        self.requests: list[EmbeddingRequest] = []
        self.response = response

    def execute(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return self.response

    def close(self) -> None:
        pass


def language_runtime(
    selected: RecordingLanguageEngine,
    other: RecordingLanguageEngine,
) -> Runtime:
    return Runtime(
        language_models={"selected": selected, "other": other},
        default_language_model="other",
    )


def embedding_runtime(
    selected: RecordingEmbeddingEngine,
    other: RecordingEmbeddingEngine,
) -> Runtime:
    return Runtime(
        embeddings={"selected": selected, "other": other},
        default_embedding="other",
    )


def examples(prompt: object) -> tuple[str, ...]:
    return tuple(prompt.value)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class RemoteCase:
    name: str
    invoke: Callable[[Runtime, Symbol[str], Symbol[str]], Symbol[object]]
    request: LanguageModelRequest
    boolean: bool = False


REMOTE_CASES = (
    RemoteCase(
        "summarize",
        lambda runtime, source, _other: text.summarize(
            runtime, source, engine="selected"
        ),
        language_request(
            "Summarize the content of the following text:\n",
            "Text: source\n",
        ),
    ),
    RemoteCase(
        "translate",
        lambda runtime, source, _other: text.translate(
            runtime, source, "French", engine="selected"
        ),
        language_request(
            "Your task is to translate and **only** translate the text into French:\n",
            "source",
        ),
    ),
    RemoteCase(
        "modify",
        lambda runtime, source, _other: text.modify(
            runtime, source, "make concise", engine="selected"
        ),
        language_request(
            "Modify the text to match the criteria:\n",
            "text 'source' modify 'make concise'=>",
            examples=examples(Modify()),
        ),
    ),
    RemoteCase(
        "filter",
        lambda runtime, source, _other: text.filter(
            runtime, source, "facts", engine="selected"
        ),
        language_request(
            "Filter the text to retain only information matching the criteria. "
            "Leave matching sentences unchanged:\n",
            "text 'source' criteria 'facts' =>",
        ),
    ),
    RemoteCase(
        "map",
        lambda runtime, source, _other: text.map(
            runtime, source, "uppercase names", engine="selected"
        ),
        language_request(
            "Transform each element in the input based on the instruction. "
            "Preserve container type and elements that don't match the instruction:\n",
            "text 'source' uppercase names =>",
            examples=examples(MapExpression()),
        ),
    ),
    RemoteCase(
        "convert",
        lambda runtime, source, _other: text.convert(
            runtime, source, "JSON", engine="selected"
        ),
        language_request(
            "Translate the following text into JSON format.\n",
            "text source format 'JSON' =>",
            examples=examples(Format()),
        ),
    ),
    RemoteCase(
        "style",
        lambda runtime, source, _other: text.style(
            runtime, source, "a compact table", engine="selected"
        ),
        language_request(
            "Style the data based on best practices and the requested description. "
            "Do not remove or invent content.\n",
            "[FORMAT]: a compact table\n[DATA]:\nsource\n",
        ),
    ),
    RemoteCase(
        "replace",
        lambda runtime, source, _other: text.replace(
            runtime, source, "old", "new", engine="selected"
        ),
        language_request(
            "Replace text parts by string pattern.\n",
            "text 'source' replace 'old' with 'new'=>",
            examples=examples(ReplaceText()),
        ),
    ),
    RemoteCase(
        "include",
        lambda runtime, source, _other: text.include(
            runtime, source, "a caveat", engine="selected"
        ),
        language_request(
            "Include information based on description.\n",
            "text 'source' include 'a caveat' =>",
            examples=examples(IncludeText()),
        ),
    ),
    RemoteCase(
        "combine",
        lambda runtime, source, other: text.combine(
            runtime, source, other, engine="selected"
        ),
        language_request(
            "Add the two data types in a logical way:\n",
            "source + other =>",
            examples=examples(CombineText()),
        ),
    ),
    RemoteCase(
        "extract",
        lambda runtime, source, _other: text.extract(
            runtime, source, "dates", engine="selected"
        ),
        language_request(
            "Extract a pattern from text:\n",
            "from 'source' extract 'dates' =>",
            examples=examples(ExtractPattern()),
        ),
    ),
    RemoteCase(
        "query",
        lambda runtime, source, _other: reason.query(
            runtime, source, "What is the thesis?", engine="selected"
        ),
        language_request(
            "Answer the question using only the provided data:\n",
            "Data:\nsource\nQuestion: What is the thesis?\nAnswer:",
        ),
    ),
    RemoteCase(
        "interpret",
        lambda runtime, source, _other: reason.interpret(
            runtime, source, engine="selected"
        ),
        language_request(
            "Evaluate the symbolic expression and return only the result:\n",
            "source =>",
            examples=examples(SimpleSymbolicExpression()),
        ),
    ),
    RemoteCase(
        "logic",
        lambda runtime, source, other: reason.logic(
            runtime, source, "AND", other, engine="selected"
        ),
        language_request(
            "Evaluate the logic expression:\n",
            "expr :source: AND :other: =>",
            examples=examples(LogicExpression()),
        ),
    ),
    RemoteCase(
        "equals",
        lambda runtime, source, other: compare.equals(
            runtime, source, other, engine="selected"
        ),
        language_request(
            "Make a fuzzy equality comparison. Are the following objects contextually the same?\n",
            "source == other =>",
            examples=examples(FuzzyEquals()),
        ),
        boolean=True,
    ),
    RemoteCase(
        "contains",
        lambda runtime, source, other: compare.contains(
            runtime, source, other, engine="selected"
        ),
        language_request(
            "Is the information in 'A' semantically contained in 'B'?\n",
            "other in source =>",
            examples=examples(ContainsValue()),
        ),
        boolean=True,
    ),
    RemoteCase(
        "is_instance_of",
        lambda runtime, source, _other: compare.is_instance_of(
            runtime, source, "a programming language", engine="selected"
        ),
        language_request(
            "Is 'A' semantically an instance of the described type 'B'?\n",
            "source is instance of a programming language =>",
            examples=examples(IsInstanceOf()),
        ),
        boolean=True,
    ),
    RemoteCase(
        "rank",
        lambda runtime, source, _other: rank.rank(
            runtime, source, "quality", engine="selected"
        ),
        language_request(
            "Rank the objects from highest to lowest by the requested measure:\n",
            "measure: 'quality' list: source =>",
            examples=examples(RankList()),
        ),
    ),
)


@pytest.mark.parametrize("case", REMOTE_CASES, ids=lambda case: case.name)
def test_language_operation_contract(
    case: RemoteCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = RecordingLanguageEngine(
        language_response("  'yes'  " if case.boolean else "  decoded  ")
    )
    other = RecordingLanguageEngine(language_response("wrong engine"))
    explicit = language_runtime(selected, other)
    ambient_engine = RecordingLanguageEngine(language_response("ambient"))
    ambient = Runtime(
        language_models={"ambient": ambient_engine},
        default_language_model="ambient",
    )
    source = Symbol("source")
    other_symbol = Symbol("other")
    function_calls: list[Function] = []
    decoders: list[object] = []
    wraps: list[object] = []
    original_function_call = Function.__call__
    original_decode = primitives.decode_output
    original_symbol = primitives.Symbol

    def record_function_call(
        function: Function,
        runtime: Runtime,
        *values: object,
        engine: str | None = None,
    ) -> LanguageModelResponse:
        function_calls.append(function)
        return original_function_call(function, runtime, *values, engine=engine)

    def record_decode(response: LanguageModelResponse, decoder: object) -> object:
        decoders.append(decoder)
        return original_decode(response, decoder)

    def record_wrap(value: object) -> Symbol[object]:
        wraps.append(value)
        return original_symbol(value)

    def fail_ambient_lookup() -> Runtime:
        raise AssertionError("ambient Runtime lookup is forbidden")

    monkeypatch.setattr(Function, "__call__", record_function_call)
    monkeypatch.setattr(primitives, "decode_output", record_decode)
    monkeypatch.setattr(primitives, "Symbol", record_wrap)
    monkeypatch.setattr(runtime_module, "current_runtime", fail_ambient_lookup)

    with explicit, ambient:
        result = case.invoke(explicit, source, other_symbol)

    assert result.value is True if case.boolean else result.value == "decoded"
    assert result is not source
    assert result is not other_symbol
    assert source.value == "source"
    assert other_symbol.value == "other"
    assert selected.requests == [case.request]
    assert other.requests == []
    assert ambient_engine.requests == []
    assert len(function_calls) == 1
    assert len(decoders) == 1
    assert len(wraps) == 1
    assert isinstance(
        decoders[0], ConstructorDecoder if case.boolean else TextDecoder
    )
    if case.boolean:
        assert decoders[0].constructor is bool  # type: ignore[union-attr]
    assert selected.response.metadata is METADATA
    assert not hasattr(result, "metadata")


def test_embedding_is_explicit_ordered_and_executes_once_without_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = EmbeddingResponse(
        vectors=(
            EmbeddingVector(index=1, values=(3.0, 4.0)),
            EmbeddingVector(index=0, values=(1.0, 2.0)),
        ),
        metadata=METADATA,
    )
    selected = RecordingEmbeddingEngine(response)
    other = RecordingEmbeddingEngine(response)
    source = Symbol(("first", "second"))
    function_calls: list[Function] = []

    def fail_function(*_args: object, **_kwargs: object) -> LanguageModelResponse:
        raise AssertionError("Embedding must not execute through Function")

    def fail_ambient_lookup() -> Runtime:
        raise AssertionError("ambient Runtime lookup is forbidden")

    monkeypatch.setattr(Function, "__call__", fail_function)
    monkeypatch.setattr(runtime_module, "current_runtime", fail_ambient_lookup)

    runtime = embedding_runtime(selected, other)
    with runtime:
        result = embed.embed(
            runtime,
            source,
            dimensions=2,
            user="tenant-user",
            engine="selected",
        )

    assert result.value == [[1.0, 2.0], [3.0, 4.0]]
    assert result is not source
    assert source.value == ("first", "second")
    assert selected.requests == [
        embedding_request(("first", "second"), dimensions=2, user="tenant-user")
    ]
    assert other.requests == []
    assert function_calls == []
    assert response.metadata is METADATA
    assert not hasattr(result, "metadata")


@pytest.mark.parametrize(
    ("inputs", "indices"),
    [
        pytest.param(("first", "second"), (0,), id="missing"),
        pytest.param(("first",), (0, 1), id="extra"),
        pytest.param(("first", "second"), (0, 0), id="duplicate"),
        pytest.param(("first", "second"), (0, 2), id="out-of-range"),
        pytest.param(("first",), (-1,), id="negative"),
    ],
)
def test_embedding_rejects_response_indices_outside_the_input_index_set(
    inputs: tuple[str, ...],
    indices: tuple[int, ...],
) -> None:
    vectors = tuple(
        EmbeddingVector.model_construct(index=index, values=(float(position),))
        for position, index in enumerate(indices)
    )
    response = EmbeddingResponse.model_construct(vectors=vectors, metadata=METADATA)
    engine = RecordingEmbeddingEngine(response)
    runtime = Runtime(embeddings={"selected": engine}, default_embedding="selected")

    with runtime, pytest.raises(ValueError, match="indices"):
        embed.embed(runtime, Symbol(inputs))

    assert engine.requests == [embedding_request(inputs)]


@pytest.mark.parametrize("value", ["", (), [], ("ok", 1), 1, b"bytes"])
def test_embedding_rejects_empty_or_non_text_symbol_values(value: object) -> None:
    engine = RecordingEmbeddingEngine(
        EmbeddingResponse(
            vectors=(EmbeddingVector(index=0, values=(0.0,)),),
            metadata=METADATA,
        )
    )
    runtime = Runtime(embeddings={"selected": engine}, default_embedding="selected")

    with runtime, pytest.raises((TypeError, ValueError), match="non-empty text"):
        embed.embed(runtime, Symbol(value))

    assert engine.requests == []


def test_template_is_local_fresh_and_non_mutating() -> None:
    source = Symbol("Ada")

    result = text.template(source, "Hello, {{name}}!", placeholder="{{name}}")

    assert result.value == "Hello, Ada!"
    assert result is not source
    assert source.value == "Ada"


@pytest.mark.parametrize(
    ("metric", "expected"),
    [("cosine", 0.0), ("dot", 0.0)],
)
def test_similarity_metrics(metric: str, expected: float) -> None:
    left = Symbol([1.0, 0.0])
    right = Symbol([0.0, 2.0])

    result = embed.similarity(left, right, metric=metric)

    assert result.value == pytest.approx(expected)
    assert result is not left
    assert left.value == [1.0, 0.0]
    assert right.value == [0.0, 2.0]


@pytest.mark.parametrize("magnitude", [1e200, 1e-200])
def test_cosine_similarity_is_stable_at_finite_numeric_extremes(
    magnitude: float,
) -> None:
    result = embed.similarity(
        Symbol([magnitude]),
        Symbol([magnitude]),
        metric="cosine",
    )

    assert result.value == pytest.approx(1.0)


def test_cosine_similarity_rejects_a_true_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        embed.similarity(Symbol([0.0, 0.0]), Symbol([1.0, 0.0]))


@pytest.mark.parametrize(
    ("metric", "options", "expected"),
    [
        ("euclidean", {}, 5.0),
        ("manhattan", {}, 7.0),
        ("minkowski", {"p": 3.0}, 4.497941445),
    ],
)
def test_distance_metrics(
    metric: str,
    options: dict[str, float],
    expected: float,
) -> None:
    result = embed.distance(
        Symbol([0.0, 0.0]),
        Symbol([3.0, 4.0]),
        metric=metric,
        **options,
    )

    assert result.value == pytest.approx(expected)


def test_minkowski_distance_requires_explicit_valid_p() -> None:
    left = Symbol([0.0])
    right = Symbol([1.0])

    with pytest.raises(ValueError, match="p"):
        embed.distance(left, right, metric="minkowski")
    with pytest.raises(ValueError, match="p"):
        embed.distance(left, right, metric="minkowski", p=0)
    with pytest.raises(ValueError, match="only valid"):
        embed.distance(left, right, metric="euclidean", p=2)


@pytest.mark.parametrize(
    ("kind", "options", "expected"),
    [
        ("linear", {}, 11.0),
        ("rbf", {"gamma": 0.5}, np.exp(-4.0)),
        ("polynomial", {"degree": 2, "coef0": 1.0}, 144.0),
    ],
)
def test_kernel_functions(
    kind: str,
    options: dict[str, float | int],
    expected: float,
) -> None:
    result = embed.kernel(
        Symbol([1.0, 2.0]),
        Symbol([3.0, 4.0]),
        kind=kind,
        **options,
    )

    assert result.value == pytest.approx(expected)


def test_kernel_options_are_mode_specific() -> None:
    left = Symbol([1.0])
    right = Symbol([1.0])

    with pytest.raises(ValueError, match="gamma"):
        embed.kernel(left, right, kind="rbf")
    with pytest.raises(ValueError, match="degree.*coef0"):
        embed.kernel(left, right, kind="polynomial", degree=2)
    with pytest.raises(ValueError, match="not valid"):
        embed.kernel(left, right, kind="linear", gamma=1.0)


def test_rbf_mmd_is_zero_for_identical_samples_and_bounded() -> None:
    samples = Symbol([[0.0, 1.0], [1.0, 0.0]])

    result = embed.mmd(samples, samples, gamma=0.5)

    assert result.value == pytest.approx(0.0)
    oversized = Symbol(np.zeros((1001, 1)))
    with pytest.raises(ValueError, match="bounded"):
        embed.mmd(oversized, oversized, gamma=1.0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: embed.similarity(Symbol([1.0]), Symbol([1.0]), metric="bad"), "metric"),
        (lambda: embed.distance(Symbol([1.0]), Symbol([1.0]), metric="bad"), "metric"),
        (lambda: embed.kernel(Symbol([1.0]), Symbol([1.0]), kind="bad"), "kind"),
    ],
)
def test_numeric_helpers_reject_unsupported_modes(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: embed.similarity(Symbol([1.0, 2.0]), Symbol([1.0])),
        lambda: embed.distance(Symbol(["1"]), Symbol([1.0])),
        lambda: embed.kernel(Symbol([]), Symbol([])),
        lambda: embed.mmd(Symbol([1.0]), Symbol([1.0]), gamma=1.0),
    ],
)
def test_numeric_helpers_propagate_useful_shape_and_numeric_errors(
    call: Callable[[], object],
) -> None:
    with pytest.raises((TypeError, ValueError), match="numeric|shape|one-dimensional|two-dimensional|non-empty"):
        call()


def test_public_namespaces_and_functions_are_exact() -> None:
    assert ops.__all__ == ("compare", "embed", "rank", "reason", "text")
    assert "primitives" not in ops.__all__
    assert text.__all__ == (
        "summarize",
        "translate",
        "modify",
        "filter",
        "map",
        "convert",
        "style",
        "template",
        "replace",
        "include",
        "combine",
        "extract",
    )
    assert reason.__all__ == ("query", "interpret", "logic")
    assert compare.__all__ == ("equals", "contains", "is_instance_of")
    assert rank.__all__ == ("rank",)
    assert embed.__all__ == ("embed", "similarity", "distance", "mmd", "kernel")


def test_remote_signatures_have_only_explicit_engine_selection() -> None:
    remote_functions = [
        text.summarize,
        text.translate,
        text.modify,
        text.filter,
        text.map,
        text.convert,
        text.style,
        text.replace,
        text.include,
        text.combine,
        text.extract,
        reason.query,
        reason.interpret,
        reason.logic,
        compare.equals,
        compare.contains,
        compare.is_instance_of,
        rank.rank,
        embed.embed,
    ]

    for operation in remote_functions:
        signature = inspect.signature(operation)
        assert tuple(signature.parameters)[0] == "runtime"
        assert signature.parameters["engine"].kind is inspect.Parameter.KEYWORD_ONLY
        assert "provider" not in signature.parameters
        assert "model" not in signature.parameters
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    for operation in (text.template, embed.similarity, embed.distance, embed.mmd, embed.kernel):
        signature = inspect.signature(operation)
        assert "runtime" not in signature.parameters
        assert "engine" not in signature.parameters


def test_raw_primary_operands_are_rejected() -> None:
    language = RecordingLanguageEngine(language_response("unused"))
    runtime = Runtime(language_models={"language": language}, default_language_model="language")

    with runtime:
        with pytest.raises(TypeError, match="Symbol"):
            text.summarize(runtime, "raw")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Symbol"):
            compare.equals(runtime, Symbol("one"), "raw")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Symbol"):
        embed.similarity(Symbol([1.0]), [1.0])  # type: ignore[arg-type]


def test_old_mixin_context_and_symbol_surfaces_are_absent() -> None:
    for name in (
        "Primitive",
        "OperatorPrimitives",
        "CastingPrimitives",
        "IterationPrimitives",
        "ValueHandlingPrimitives",
        "StringHelperPrimitives",
        "ComparisonPrimitives",
        "ExpressionHandlingPrimitives",
        "DataHandlingPrimitives",
        "PatternMatchingPrimitives",
        "QueryHandlingPrimitives",
        "TemplateStylingPrimitives",
        "EmbeddingPrimitives",
        "PersistencePrimitives",
        "current_runtime",
        "contextualize_language_request",
    ):
        assert not hasattr(primitives, name)

    assert not hasattr(symbol_module, "Expression")
    value = Symbol("text")
    for name in (
        "summarize",
        "translate",
        "modify",
        "filter",
        "map",
        "convert",
        "style",
        "template",
        "replace",
        "include",
        "combine",
        "extract",
        "query",
        "interpret",
        "logic",
        "equals",
        "contains",
        "isinstanceof",
        "rank",
        "embed",
        "similarity",
        "distance",
        "save",
        "load",
    ):
        assert not hasattr(value, name)
