import ast
import builtins
import operator
import socket
import sys
import urllib.request
from pathlib import Path
from typing import get_args, get_origin

import pytest

import symai.symbol as symbol_module
from symai.symbol import Symbol


def test_symbol_supports_typing_subscription() -> None:
    alias = Symbol[str | list[str]]

    assert get_origin(alias) is Symbol
    assert get_args(alias) == (str | list[str],)


def test_symbol_holds_exactly_the_caller_owned_reference() -> None:
    held = [{"nested": []}]
    symbol = Symbol(held)

    assert symbol.value is held
    assert Symbol(symbol).value is symbol

    held[0]["nested"].append("changed")

    assert symbol.value == [{"nested": ["changed"]}]


def test_symbol_wrapper_state_is_immutable_and_cannot_be_extended() -> None:
    symbol = Symbol("original")

    for name, value in (("_value", "changed"), ("value", "changed"), ("extra", 1)):
        with pytest.raises(AttributeError):
            setattr(symbol, name, value)

    for name in ("_value", "value"):
        with pytest.raises(AttributeError):
            delattr(symbol, name)

    assert symbol.value == "original"
    assert not hasattr(symbol, "__dict__")


def test_symbol_is_unhashable_even_when_its_value_is_hashable() -> None:
    with pytest.raises(TypeError):
        hash(Symbol("hashable"))


def test_equality_is_symmetric_for_raw_and_symbol_operands() -> None:
    raw = [1, 2]
    symbol = Symbol(raw)

    assert symbol == raw
    assert raw == symbol
    assert symbol == Symbol([1, 2])
    assert symbol != [2, 1]
    assert symbol != Symbol([2, 1])
    assert symbol.__eq__(raw) is True
    assert symbol.__ne__([2, 1]) is True


def test_ordering_returns_native_booleans_and_unwraps_symbol_operands() -> None:
    symbol = Symbol(2)

    assert symbol < 3
    assert symbol <= Symbol(2)
    assert symbol > 1
    assert symbol >= Symbol(2)
    assert symbol > 1
    assert symbol < 3
    assert symbol.__lt__(3) is True
    assert symbol.__le__(Symbol(2)) is True
    assert symbol.__gt__(1) is True
    assert symbol.__ge__(Symbol(2)) is True


def test_containment_uses_the_held_value_as_container() -> None:
    container = Symbol(["alpha", "beta"])

    assert "alpha" in container
    assert Symbol("alpha") in container
    assert Symbol("alpha") in ["alpha", "beta"]

    with pytest.raises(TypeError) as error:
        _ = container in Symbol("alpha")

    assert error.value.args == ("'in <string>' requires string as left operand, not list",)


@pytest.mark.parametrize(
    ("operation", "left", "right", "expected"),
    [
        (operator.add, 8, 2, 10),
        (operator.sub, 8, 2, 6),
        (operator.mul, 8, 2, 16),
        (operator.truediv, 8, 2, 4.0),
        (operator.floordiv, 9, 2, 4),
        (operator.mod, 9, 2, 1),
        (operator.pow, 3, 2, 9),
        (divmod, 9, 2, (4, 1)),
    ],
)
def test_binary_arithmetic_returns_new_symbols(
    operation: object,
    left: int,
    right: int,
    expected: object,
) -> None:
    apply = operation
    left_symbol = Symbol(left)

    result = apply(left_symbol, Symbol(right))

    assert isinstance(result, Symbol)
    assert result is not left_symbol
    assert result.value == expected
    assert left_symbol.value == left


@pytest.mark.parametrize(
    ("operation", "left", "right", "expected"),
    [
        (operator.add, 8, 2, 10),
        (operator.sub, 8, 2, 6),
        (operator.mul, 8, 2, 16),
        (operator.truediv, 8, 2, 4.0),
        (operator.floordiv, 9, 2, 4),
        (operator.mod, 9, 2, 1),
        (operator.pow, 3, 2, 9),
        (divmod, 9, 2, (4, 1)),
    ],
)
def test_reflected_arithmetic_returns_new_symbols(
    operation: object,
    left: int,
    right: int,
    expected: object,
) -> None:
    apply = operation
    right_symbol = Symbol(right)

    result = apply(left, right_symbol)

    assert isinstance(result, Symbol)
    assert result is not right_symbol
    assert result.value == expected
    assert right_symbol.value == right


def test_power_supports_the_native_three_argument_form() -> None:
    result = pow(Symbol(5), Symbol(3), Symbol(13))

    assert isinstance(result, Symbol)
    assert result.value == pow(5, 3, 13)


def test_reflected_power_supports_the_native_three_argument_form() -> None:
    direct = Symbol(3).__rpow__(2, Symbol(5))

    assert isinstance(direct, Symbol)
    assert direct.value == pow(2, 3, 5)

    if sys.version_info >= (3, 14):
        reflected = pow(2, Symbol(3), Symbol(5))

        assert isinstance(reflected, Symbol)
        assert reflected.value == pow(2, 3, 5)
    else:
        with pytest.raises(TypeError):
            pow(2, Symbol(3), Symbol(5))


class MatrixValue:
    def __init__(self, value: str) -> None:
        self.value = value

    def __matmul__(self, other: object) -> object:
        if not isinstance(other, MatrixValue):
            return NotImplemented

        return f"{self.value}@{other.value}"

    def __rmatmul__(self, other: "MatrixValue") -> str:
        return f"{other.value}@{self.value}"


def test_matrix_multiplication_and_its_reflection_return_symbols() -> None:
    left = MatrixValue("left")
    right = MatrixValue("right")

    direct = Symbol(left) @ Symbol(right)
    reflected = left @ Symbol(right)

    assert isinstance(direct, Symbol)
    assert direct.value == "left@right"
    assert isinstance(reflected, Symbol)
    assert reflected.value == "left@right"


@pytest.mark.parametrize(
    ("operation", "left", "right", "expected"),
    [
        (operator.and_, 0b1100, 0b1010, 0b1000),
        (operator.or_, 0b1100, 0b1010, 0b1110),
        (operator.xor, 0b1100, 0b1010, 0b0110),
        (operator.lshift, 3, 2, 12),
        (operator.rshift, 12, 2, 3),
    ],
)
def test_bitwise_operations_and_reflections_return_symbols(
    operation: object,
    left: int,
    right: int,
    expected: int,
) -> None:
    apply = operation

    direct = apply(Symbol(left), Symbol(right))
    reflected = apply(left, Symbol(right))

    assert isinstance(direct, Symbol)
    assert direct.value == expected
    assert isinstance(reflected, Symbol)
    assert reflected.value == expected


@pytest.mark.parametrize(
    ("operation", "value", "expected"),
    [
        (operator.neg, 3, -3),
        (operator.pos, -3, -3),
        (operator.abs, -3, 3),
        (operator.invert, 3, ~3),
    ],
)
def test_unary_value_operations_return_symbols(
    operation: object,
    value: int,
    expected: int,
) -> None:
    apply = operation

    result = apply(Symbol(value))

    assert isinstance(result, Symbol)
    assert result.value == expected


def test_indexing_and_slicing_return_new_symbols() -> None:
    held = [{"id": 1}, {"id": 2}, {"id": 3}]
    symbol = Symbol(held)

    indexed = symbol[Symbol(1)]
    sliced = symbol[1:]

    assert isinstance(indexed, Symbol)
    assert indexed.value is held[1]
    assert isinstance(sliced, Symbol)
    assert sliced.value == held[1:]
    assert sliced.value is not held


def test_iteration_yields_symbols_without_copying_elements() -> None:
    first = {"id": 1}
    second = {"id": 2}
    held = [first, second]

    values = list(Symbol(held))

    assert all(isinstance(value, Symbol) for value in values)
    assert [value.value for value in values] == held
    assert values[0].value is first
    assert values[1].value is second


def test_len_truth_and_explicit_casts_return_native_values() -> None:
    sequence = Symbol([1, 2])

    assert len(sequence) == 2
    assert bool(sequence) is True
    assert bool(Symbol([])) is False
    assert str(Symbol(12)) == "12"
    assert int(Symbol("12")) == 12
    assert float(Symbol("1.5")) == 1.5


def test_native_exceptions_propagate_unchanged() -> None:
    with pytest.raises(TypeError) as native_type_error:
        _ = 1 + "x"
    with pytest.raises(TypeError) as symbol_type_error:
        _ = Symbol(1) + "x"

    with pytest.raises(KeyError) as native_key_error:
        _ = {"present": 1}["missing"]
    with pytest.raises(KeyError) as symbol_key_error:
        _ = Symbol({"present": 1})["missing"]

    with pytest.raises(IndexError) as native_index_error:
        _ = [1][2]  # noqa: PLE0643  # the IndexError is the behaviour under comparison
    with pytest.raises(IndexError) as symbol_index_error:
        _ = Symbol([1])[2]

    assert symbol_type_error.value.args == native_type_error.value.args
    assert symbol_key_error.value.args == native_key_error.value.args
    assert symbol_index_error.value.args == native_index_error.value.args


def test_symbol_exposes_no_item_or_in_place_mutation() -> None:
    symbol = Symbol([1, 2])

    with pytest.raises(TypeError):
        symbol[0] = 3
    with pytest.raises(TypeError):
        del symbol[0]

    in_place_dunders = {
        "__iadd__",
        "__isub__",
        "__imul__",
        "__imatmul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__iand__",
        "__ior__",
        "__ixor__",
        "__ilshift__",
        "__irshift__",
    }
    assert in_place_dunders.isdisjoint(Symbol.__dict__)

    original = symbol
    symbol += [3]

    assert symbol is not original
    assert original.value == [1, 2]
    assert symbol.value == [1, 2, 3]


def test_symbol_does_not_forward_the_held_objects_api() -> None:
    symbol = Symbol("value")

    with pytest.raises(AttributeError):
        _ = symbol.upper

    assert "__getattr__" not in Symbol.__dict__


def test_symbol_has_no_forbidden_god_object_surface() -> None:
    forbidden = {
        "_semantic",
        "sem",
        "syn",
        "static_context",
        "dynamic_context",
        "global_context",
        "parent",
        "children",
        "root",
        "nodes",
        "edges",
        "linker",
        "embedding",
        "adapt",
        "clear",
        "save",
        "load",
        "json",
        "_to_symbol",
        "_to_type",
        "__setitem__",
        "__delitem__",
    }

    assert forbidden.isdisjoint(Symbol.__dict__)


def test_symbol_module_has_no_execution_or_io_dependencies() -> None:
    source = Path(symbol_module.__file__).read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert imported_modules.isdisjoint(
        {
            "httpx",
            "requests",
            "socket",
            "urllib",
            "symai",
        }
    )
    assert "open" not in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def test_native_symbol_operations_do_not_perform_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_io(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a native Symbol operation attempted I/O")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", unexpected_io)
        patch.setattr(socket, "socket", unexpected_io)
        patch.setattr(urllib.request, "urlopen", unexpected_io)

        assert (Symbol(2) + Symbol(3)).value == 5
        assert Symbol("value") == "value"
        assert Symbol(["value"])[0].value == "value"
        assert str(Symbol("value")) == "value"
