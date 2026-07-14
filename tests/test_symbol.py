import pickle
from typing import get_args, get_origin

import numpy as np
import pytest

from symai.symbol import Expression, Symbol


def test_symbol_supports_typing_subscription():
    alias = Symbol[str | list[str]]
    assert get_origin(alias) is Symbol
    assert get_args(alias) == (str | list[str],)


def test_nested_symbol_preserves_explicit_context_and_graph_state() -> None:
    inner = Symbol(
        "payload",
        static_context="stable",
        dynamic_context="dynamic",
        semantic=True,
    )

    outer = Symbol(inner)

    assert outer.value == "payload"
    assert outer.static_context == "stable"
    assert outer.dynamic_context == "dynamic"
    assert outer.children == [inner]
    assert inner.parent is outer
    assert inner.root is outer
    assert outer.root is outer
    assert not hasattr(outer, "metadata")
    assert not hasattr(inner, "metadata")


def test_symbol_unwraps_nested_structures():
    nested = [
        Symbol("alpha"),
        {"k": Symbol("beta")},
        (Symbol("gamma"),),
        {Symbol("delta")},
    ]

    symbol = Symbol(nested)

    assert symbol.value == ["alpha", {"k": "beta"}, ("gamma",), {"delta"}]


def test_symbol_dynamic_context_adapt_and_clear():
    symbol = Symbol("value")
    type_key = str(type(symbol))
    original_context = Symbol._dynamic_context.get(type_key)
    original_copy = list(original_context) if original_context is not None else None

    try:
        symbol.clear(type(symbol))
        symbol.adapt("first")
        symbol.adapt(Symbol("second"))

        context_lines = symbol.dynamic_context.strip().splitlines()
        assert context_lines == ["first", "second"]

        symbol.clear(type(symbol))
        assert symbol.dynamic_context == ""
    finally:
        if original_copy is None:
            Symbol._dynamic_context.pop(type_key, None)
        else:
            Symbol._dynamic_context[type_key] = original_copy


def test_symbol_nodes_and_edges_cover_hierarchy():
    leaf = Symbol("leaf")
    middle = Symbol(leaf)
    root = Symbol(middle)

    nodes = root.nodes
    assert nodes[0] is root
    assert middle in nodes
    assert leaf in nodes

    edges = root.edges
    assert (root, middle) in edges
    assert (middle, leaf) in edges
    assert leaf.root is root


def test_symbol_json_excludes_internal_relationships():
    symbol = Symbol("value")
    serialized = symbol.json()

    assert "_metadata" not in serialized
    assert "_parent" not in serialized
    assert "_children" not in serialized
    assert serialized["_value"] == "value"


def test_symbol_pickle_round_trip_preserves_explicit_runtime_state() -> None:
    symbol = Symbol(
        "value",
        static_context="stable",
        dynamic_context="dynamic",
        semantic=True,
    )

    restored = pickle.loads(pickle.dumps(symbol))

    assert restored.value == "value"
    assert restored.static_context == "stable"
    assert restored.dynamic_context == "dynamic"
    assert (restored + " suffix").value == "value suffix"


def test_symbol_embedding_cache_is_explicit_and_reused() -> None:
    symbol = Symbol([1.0, 2.0])

    first = symbol.embedding
    second = symbol.embedding

    assert isinstance(first, np.ndarray)
    assert second is first


def test_symbol_rejects_dynamic_callable_attachment() -> None:
    with pytest.raises(TypeError, match="callables"):
        Symbol("value", callables=[("dynamic", lambda symbol: symbol)])


def test_string_helpers_raise_explicit_type_errors() -> None:
    with pytest.raises(TypeError, match="delimiter must be a string"):
        Symbol("value").split(1)
    with pytest.raises(TypeError, match="value must be a string"):
        Symbol(1).startswith("1")


@pytest.mark.parametrize(
    ("symbol", "method_name", "args"),
    [
        (Symbol("value"), "startswith", ("v",)),
        (Symbol("value"), "endswith", ("e",)),
        (Symbol("value"), "split", ("a",)),
        (Symbol(("a", "b")), "join", (",",)),
        (Symbol("value"), "template", ("<{{placeholder}}>",)),
    ],
)
def test_local_string_primitives_reject_unknown_keyword_options(
    symbol: Symbol,
    method_name: str,
    args: tuple[str, ...],
) -> None:
    method = getattr(symbol, method_name)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        method(*args, provider="openai")


def test_empty_embedding_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        _ = Symbol([]).embedding


def test_comparison_dunders_return_booleans() -> None:
    symbol = Symbol(1)

    assert symbol.__eq__(1) is True
    assert symbol.__ne__(1) is False
    assert symbol.__lt__(2) is True
    assert symbol.__le__(1) is True
    assert symbol.__gt__(0) is True
    assert symbol.__ge__(1) is True


def test_symbol_uses_one_fixed_primitive_composition() -> None:
    assert type(Symbol("first")) is Symbol
    assert type(Symbol("second")) is Symbol


def test_expression_results_link_to_the_explicit_graph_root() -> None:
    class Echo(Expression):
        def forward(self, *_args: object, **_kwargs: object) -> Symbol:
            return Symbol("result")

    expression = Echo()
    root = Symbol(expression)

    result = expression()

    assert result.value == "result"
    assert root.linker is not None
    assert [value.value for value in root.linker.values()] == ["result"]


def test_symbol_to_symbol_preserves_context():
    symbol = Symbol("source", static_context="ctx")
    new_symbol = symbol._to_symbol("target")

    assert isinstance(new_symbol, Symbol)
    assert new_symbol.static_context == "ctx"
    assert new_symbol.value == "target"


def test_symbol_to_type_constructs_subclass_instances():
    class DerivedSymbol(Symbol[str]):
        pass

    derived = DerivedSymbol("value", static_context="ctx")
    new_instance = derived._to_type("other")

    assert isinstance(new_instance, DerivedSymbol)
    assert new_instance.static_context == "ctx"
    assert new_instance.value == "other"
