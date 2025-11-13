from typing import List, Union, get_args, get_origin

from symai.symbol import Symbol


def test_symbol_supports_typing_subscription():
    alias = Symbol[Union[str, List[str]]]
    assert get_origin(alias) is Symbol
    assert get_args(alias) == (Union[str, List[str]],)


def test_symbol_unwraps_nested_symbol_and_links_parent_child():
    inner = Symbol("payload", static_context="ctx")
    inner.metadata.custom_flag = True

    outer = Symbol(inner)

    assert outer.value == "payload"
    assert outer.static_context == inner.static_context
    assert outer.metadata is inner.metadata
    assert outer.children == [inner]
    assert inner.parent is outer
    assert inner.root is outer
    assert outer.root is outer


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


def test_symbol_to_symbol_preserves_kwargs():
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
