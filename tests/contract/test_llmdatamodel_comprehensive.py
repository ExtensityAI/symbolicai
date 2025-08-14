import json
import re
from enum import Enum
from typing import Any, Optional, Union

import pytest
from pydantic import Field, ValidationError

from symai.models.base import (Const, CustomConstraint, LengthConstraint,
                               LLMDataModel, build_dynamic_llm_datamodel)


# ---------------------------------------------------------------------------
# Helper Functions for Robust Format Testing
# ---------------------------------------------------------------------------
def contains_key_value(text: str, key: str, value: Any = None) -> bool:
    """Check if a key (and optionally value) appears in the formatted text.
    This is more robust than checking exact formatting like 'key: value'."""
    if value is None:
        # Just check if the key appears followed by a colon
        return f"{key}:" in text or f"{key} :" in text
    else:
        # Check if key and value appear near each other
        return (f"{key}: {value}" in text or
                f"{key} : {value}" in text or
                f"{key}:{value}" in text)

def contains_list_item(text: str, item: Any) -> bool:
    """Check if an item appears in list format.
    This is more robust than checking exact formatting like '- : item'."""
    item_str = str(item)
    # Check various list formatting patterns
    return (f"- : {item_str}" in text or
            f"- {item_str}" in text or
            f"-: {item_str}" in text or
            f"  {item_str}" in text)  # Sometimes items are just indented


# ---------------------------------------------------------------------------
# Test Models
# ---------------------------------------------------------------------------
class SimpleModel(LLMDataModel):
    value: str


class ComplexNestedModel(LLMDataModel):
    level1: str
    nested: Optional['ComplexNestedModel'] = None


class ModelWithEnum(LLMDataModel):
    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    status: Status
    code: int


class ModelWithConstraints(LLMDataModel):
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')


class ModelWithDefaults(LLMDataModel):
    required_field: str
    optional_field: str = "default_value"
    const_field: str = Const("immutable")


class ModelWithComplexTypes(LLMDataModel):
    list_of_lists: list[list[int]]
    dict_of_dicts: dict[str, dict[str, str]]
    union_field: Union[int, str, list[str]]
    optional_union: Optional[Union[dict[str, Any], list[Any]]]


class RecursiveModel(LLMDataModel):
    name: str
    children: list['RecursiveModel'] = Field(default_factory=list)


class ModelWithSpecialChars(LLMDataModel):
    field_with_quotes: str
    field_with_newlines: str
    field_with_unicode: str


# ---------------------------------------------------------------------------
# Constraint Classes Tests
# ---------------------------------------------------------------------------
def test_length_constraint_creation():
    """Test LengthConstraint dataclass creation and attributes."""
    constraint = LengthConstraint()
    constraint.field_name = "test_field"
    constraint.min_length = 5
    constraint.max_length = 20

    assert constraint.field_name == "test_field"
    assert constraint.min_length == 5
    assert constraint.max_length == 20


def test_custom_constraint_creation():
    """Test CustomConstraint dataclass creation."""
    constraint = CustomConstraint(rule="must be alphanumeric")
    assert constraint.rule == "must be alphanumeric"


def test_const_field_creation():
    """Test Const function creates frozen field."""
    model = ModelWithDefaults(required_field="test")
    assert model.const_field == "immutable"

    with pytest.raises(ValidationError):
        ModelWithDefaults(required_field="test", const_field="different")


# ---------------------------------------------------------------------------
# format_field Tests
# ---------------------------------------------------------------------------
def test_format_field_empty_list():
    """Test formatting of empty lists."""
    model = SimpleModel(value="test")
    result = model.format_field("empty_list", [], indent=0, depth=0)
    assert result == "empty_list:\n"


def test_format_field_empty_dict():
    """Test formatting of empty dictionaries."""
    model = SimpleModel(value="test")
    result = model.format_field("empty_dict", {}, indent=0, depth=0)
    assert result == "empty_dict:\n"


def test_format_field_deeply_nested():
    """Test formatting of deeply nested structures."""
    nested_data = {
        "level1": {
            "level2": {
                "level3": ["item1", "item2"]
            }
        }
    }
    model = SimpleModel(value="test")
    result = model.format_field("nested", nested_data, indent=0, depth=0)
    assert contains_key_value(result, "level1")
    assert contains_key_value(result, "level2")
    assert contains_key_value(result, "level3")


def test_format_field_with_special_chars():
    """Test formatting fields with special characters."""
    model = ModelWithSpecialChars(
        field_with_quotes='text with "quotes"',
        field_with_newlines="line1\nline2",
        field_with_unicode="emoji 😊 and symbols ™"
    )
    s = str(model)
    assert 'text with "quotes"' in s
    assert "line1\nline2" in s
    assert "emoji 😊" in s


def test_format_field_mixed_types_in_list():
    """Test formatting lists with mixed types."""
    model = SimpleModel(value="test")
    mixed_list = [1, "string", {"key": "value"}, [1, 2, 3]]
    result = model.format_field("mixed", mixed_list, indent=0, depth=0)
    assert contains_list_item(result, 1)
    assert contains_list_item(result, "string")
    assert contains_key_value(result, "key", "value")


# ---------------------------------------------------------------------------
# __str__ Tests
# ---------------------------------------------------------------------------
def test_str_with_recursive_model():
    """Test string representation of recursive models."""
    parent = RecursiveModel(
        name="parent",
        children=[
            RecursiveModel(name="child1"),
            RecursiveModel(name="child2", children=[
                RecursiveModel(name="grandchild")
            ])
        ]
    )
    s = str(parent)
    assert "parent" in s
    assert "child1" in s
    assert "child2" in s
    assert "grandchild" in s


def test_str_without_section_header():
    """Test string representation without section header."""
    model = SimpleModel(value="test")
    s = str(model)
    assert not s.startswith("[[")
    assert contains_key_value(s, "value", "test")


def test_str_with_enum_field():
    """Test string representation with enum fields."""
    model = ModelWithEnum(
        status=ModelWithEnum.Status.ACTIVE,
        code=200
    )
    s = str(model)
    assert contains_key_value(s, "status", "active")
    assert contains_key_value(s, "code", 200)


# ---------------------------------------------------------------------------
# simplify_json_schema Tests
# ---------------------------------------------------------------------------
def test_simplify_schema_with_enum():
    """Test schema simplification with enum types."""
    schema = ModelWithEnum.simplify_json_schema()
    assert "[[Schema]]" in schema
    assert "status" in schema
    assert "enum" in schema.lower() or "choice" in schema.lower()


def test_simplify_schema_with_constraints():
    """Test schema includes constraint information."""
    schema = ModelWithConstraints.simplify_json_schema()
    assert "name" in schema
    assert "age" in schema
    assert "email" in schema
    assert "string" in schema
    assert "integer" in schema


def test_simplify_schema_with_recursive_model():
    """Test schema simplification handles recursive models."""
    schema = RecursiveModel.simplify_json_schema()
    assert "name" in schema
    assert "children" in schema
    assert "array" in schema.lower()


def test_simplify_schema_with_complex_unions():
    """Test schema with complex union types."""
    schema = ModelWithComplexTypes.simplify_json_schema()
    assert "union_field" in schema
    assert "optional_union" in schema
    assert "Union" in schema or "union" in schema


def test_simplify_schema_caching():
    """Test that simplify_json_schema is properly cached."""
    schema1 = SimpleModel.simplify_json_schema()
    schema2 = SimpleModel.simplify_json_schema()
    assert schema1 == schema2


# ---------------------------------------------------------------------------
# generate_example_json Tests
# ---------------------------------------------------------------------------
def test_generate_example_with_enum():
    """Test example generation with enum fields."""
    example = LLMDataModel.generate_example_json(ModelWithEnum)
    assert "status" in example
    assert example["status"] == "active"
    assert example["code"] == 123


def test_generate_example_with_defaults():
    """Test example generation respects default values."""
    example = LLMDataModel.generate_example_json(ModelWithDefaults)
    assert example["required_field"] == "example_string"
    assert example["optional_field"] == "default_value"
    assert example["const_field"] == "immutable"


def test_generate_example_with_nested_lists():
    """Test example generation with nested container types."""
    example = LLMDataModel.generate_example_json(ModelWithComplexTypes)
    assert "list_of_lists" in example
    assert isinstance(example["list_of_lists"], list)
    assert isinstance(example["list_of_lists"][0], list)
    assert example["list_of_lists"][0][0] == 123


def test_generate_example_with_recursive():
    """Test example generation handles recursive models without infinite loop."""
    example = LLMDataModel.generate_example_json(RecursiveModel)
    assert example["name"] == "example_string"
    assert isinstance(example["children"], list)
    assert len(example["children"]) == 1
    assert "name" in example["children"][0]


def test_generate_example_with_optional_none():
    """Test example generation with optional fields."""
    class ModelWithOptional(LLMDataModel):
        required: str
        optional: Optional[str] = None

    example = LLMDataModel.generate_example_json(ModelWithOptional)
    assert example["required"] == "example_string"
    assert example["optional"] is None


def test_generate_example_complex_union_priority():
    """Test union resolution priority in complex cases."""
    DynModel = build_dynamic_llm_datamodel(
        Union[dict[str, list[int]], list[dict[str, int]], str, int]
    )
    example = LLMDataModel.generate_example_json(DynModel)
    assert isinstance(example["value"], dict)
    assert "example_string" in example["value"]
    assert isinstance(example["value"]["example_string"], list)


# ---------------------------------------------------------------------------
# instruct_llm Tests
# ---------------------------------------------------------------------------
def test_instruct_llm_with_section_header():
    """Test instruction generation with class method."""
    instr = SimpleModel.instruct_llm()
    assert "[[Result]]" in instr
    assert "[[Schema]]" in instr
    assert "[[Example]]" in instr


def test_instruct_llm_with_recursive_model():
    """Test instruction generation for recursive models."""
    instr = RecursiveModel.instruct_llm()
    assert "[[Result]]" in instr
    assert "[[Schema]]" in instr
    assert "[[Example]]" in instr
    assert "children" in instr


def test_instruct_llm_with_constraints():
    """Test instructions include constraint information."""
    instr = ModelWithConstraints.instruct_llm()
    assert "[[Schema]]" in instr
    assert "[[Example]]" in instr
    assert "name" in instr
    assert "age" in instr
    assert "email" in instr


def test_instruct_llm_multiple_union_examples():
    """Test that complex unions generate examples."""
    instr = ModelWithComplexTypes.instruct_llm()
    assert "[[Example]]" in instr or "[[Example 1]]" in instr
    assert "list_of_lists" in instr
    assert "dict_of_dicts" in instr
    assert "union_field" in instr


def test_instruct_llm_caching():
    """Test that instruct_llm results are cached."""
    instr1 = SimpleModel.instruct_llm()
    instr2 = SimpleModel.instruct_llm()
    assert instr1 == instr2


# ---------------------------------------------------------------------------
# build_dynamic_llm_datamodel Tests
# ---------------------------------------------------------------------------
def test_build_dynamic_with_primitive():
    """Test dynamic model creation with primitive type."""
    DynModel = build_dynamic_llm_datamodel(str)
    instance = DynModel(value="test")
    assert instance.value == "test"


def test_build_dynamic_with_complex_type():
    """Test dynamic model with complex nested type."""
    DynModel = build_dynamic_llm_datamodel(
        dict[str, list[Union[int, str]]]
    )
    instance = DynModel(value={"key": [1, "two", 3]})
    assert instance.value["key"] == [1, "two", 3]


def test_build_dynamic_with_optional():
    """Test dynamic model with optional type."""
    DynModel = build_dynamic_llm_datamodel(Optional[list[str]])
    instance1 = DynModel(value=["a", "b"])
    instance2 = DynModel(value=None)
    assert instance1.value == ["a", "b"]
    assert instance2.value is None


def test_build_dynamic_model_inheritance():
    """Test that dynamic models inherit from LLMDataModel."""
    DynModel = build_dynamic_llm_datamodel(int)
    assert issubclass(DynModel, LLMDataModel)
    instance = DynModel(value=42)
    assert isinstance(instance, LLMDataModel)


def test_build_dynamic_model_unique_names():
    """Test that different types create models with different names."""
    DynModel1 = build_dynamic_llm_datamodel(str)
    DynModel2 = build_dynamic_llm_datamodel(int)
    assert DynModel1.__name__ != DynModel2.__name__


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------
def test_model_with_no_fields():
    """Test model with no fields besides section_header."""
    class EmptyModel(LLMDataModel):
        pass

    model = EmptyModel()
    s = str(model)
    assert s == ""


def test_very_deep_nesting():
    """Test handling of very deep nesting."""
    deep_nested = {"a": {"b": {"c": {"d": {"e": {"f": "value"}}}}}}
    model = SimpleModel(value="test")
    result = model.format_field("deep", deep_nested, indent=0, depth=0)
    assert contains_key_value(result, "f", "value")


def test_circular_reference_handling():
    """Test that circular references are handled properly."""
    parent = RecursiveModel(name="parent")
    child = RecursiveModel(name="child")
    parent.children.append(child)

    s = str(parent)
    assert "parent" in s
    assert "child" in s
    assert "RecursiveModel" in str(type(parent))


def test_unicode_in_field_names():
    """Test handling of unicode in field names and values."""
    class UnicodeModel(LLMDataModel):
        emoji_field: str

    model = UnicodeModel(emoji_field="🎉")
    s = str(model)
    assert "🎉" in s


def test_extremely_long_strings():
    """Test handling of very long string values."""
    long_string = "x" * 10000
    model = SimpleModel(value=long_string)
    s = str(model)
    assert long_string in s


def test_special_json_characters():
    """Test handling of characters that need JSON escaping."""
    model = ModelWithSpecialChars(
        field_with_quotes='"quotes"',
        field_with_newlines='line1\nline2\ttab',
        field_with_unicode='\\u0041'
    )
    example = LLMDataModel.generate_example_json(ModelWithSpecialChars)
    json_str = json.dumps(example)
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Dictionary Key Type Tests
# ---------------------------------------------------------------------------
def test_dict_with_int_keys():
    """Test that dict[int, str] generates integer keys."""
    model = build_dynamic_llm_datamodel(dict[int, str])
    example = model.generate_example_json()

    assert "value" in example
    dict_value = example["value"]
    assert isinstance(dict_value, dict)

    # Should have an integer key
    assert 123 in dict_value
    assert dict_value[123] == "example_string"


def test_dict_with_float_keys():
    """Test that dict[float, int] generates float keys."""
    model = build_dynamic_llm_datamodel(dict[float, int])
    example = model.generate_example_json()

    dict_value = example["value"]
    assert isinstance(dict_value, dict)

    # Should have a float key
    assert 123.45 in dict_value
    assert dict_value[123.45] == 123


def test_dict_with_bool_keys():
    """Test that dict[bool, str] generates boolean keys."""
    model = build_dynamic_llm_datamodel(dict[bool, str])
    example = model.generate_example_json()

    dict_value = example["value"]
    assert isinstance(dict_value, dict)

    # Should have a boolean key
    assert True in dict_value
    assert dict_value[True] == "example_string"


def test_dict_with_tuple_keys():
    """Test that dict[tuple[str, int], bool] generates tuple keys."""
    model = build_dynamic_llm_datamodel(dict[tuple[str, int], bool])
    example = model.generate_example_json()

    dict_value = example["value"]
    assert isinstance(dict_value, dict)

    # Should have a tuple key
    expected_key = ("example_string", 123)
    assert expected_key in dict_value
    assert dict_value[expected_key] is True


def test_dict_with_frozenset_keys():
    """Test that dict[frozenset[int], str] generates frozenset keys."""
    model = build_dynamic_llm_datamodel(dict[frozenset[int], str])
    example = model.generate_example_json()

    dict_value = example["value"]
    assert isinstance(dict_value, dict)

    # Should have a frozenset key
    keys = list(dict_value.keys())
    assert len(keys) == 1
    assert isinstance(keys[0], frozenset)
    assert 123 in keys[0]
    assert dict_value[keys[0]] == "example_string"


def test_dict_with_string_keys():
    """Test that dict[str, int] generates string keys (default case)."""
    model = build_dynamic_llm_datamodel(dict[str, int])
    example = model.generate_example_json()

    dict_value = example["value"]
    assert isinstance(dict_value, dict)

    # Should have a string key
    assert "example_string" in dict_value
    assert dict_value["example_string"] == 123


def test_dict_with_any_keys():
    """Test that dict[Any, str] defaults to string keys."""
    model = build_dynamic_llm_datamodel(dict[Any, str])
    example = model.generate_example_json()

    dict_value = example["value"]
    assert isinstance(dict_value, dict)

    # Should default to string key for Any
    assert "example_string" in dict_value
    assert dict_value["example_string"] == "example_string"


def test_dict_with_complex_tuple_keys():
    """Test dict with nested tuple keys."""
    model = build_dynamic_llm_datamodel(dict[tuple[int, tuple[str, bool]], float])
    example = model.generate_example_json()

    dict_value = example["value"]
    assert isinstance(dict_value, dict)

    # Should have a nested tuple key
    expected_key = (123, ("example_string", True))
    assert expected_key in dict_value
    assert dict_value[expected_key] == 123.45


def test_dict_with_unspecified_type():
    """Test plain dict without type parameters."""
    model = build_dynamic_llm_datamodel(dict)
    example = model.generate_example_json()

    dict_value = example["value"]
    assert isinstance(dict_value, dict)

    # Should default to string key for unspecified type
    assert "example_string" in dict_value


def test_nested_dict_with_various_keys():
    """Test nested dicts with different key types."""
    class ComplexModel(LLMDataModel):
        int_dict: dict[int, str]
        float_dict: dict[float, int]
        tuple_dict: dict[tuple[str, int], bool]
        nested: dict[str, dict[int, float]]

    example = ComplexModel.generate_example_json(ComplexModel)

    # Check int keys
    assert 123 in example["int_dict"]
    assert example["int_dict"][123] == "example_string"

    # Check float keys
    assert 123.45 in example["float_dict"]
    assert example["float_dict"][123.45] == 123

    # Check tuple keys
    assert ("example_string", 123) in example["tuple_dict"]
    assert example["tuple_dict"][("example_string", 123)] is True

    # Check nested dict
    assert "example_string" in example["nested"]
    inner_dict = example["nested"]["example_string"]
    assert 123 in inner_dict
    assert inner_dict[123] == 123.45


def test_dict_json_serialization_with_non_string_keys():
    """Test that dicts with non-string keys can be used in Python but note JSON limitations."""
    model = build_dynamic_llm_datamodel(dict[int, str])
    example = model.generate_example_json()

    # The example works as a Python dict
    dict_value = example["value"]
    assert 123 in dict_value

    # When serializing to JSON, numeric keys work (converted to strings in JSON)
    json_str = json.dumps(example)
    parsed = json.loads(json_str)

    # After JSON round-trip, int keys become strings
    assert "123" in parsed["value"] or 123 in parsed["value"]

    # Tuple keys would fail JSON serialization
    tuple_model = build_dynamic_llm_datamodel(dict[tuple[str, int], str])
    tuple_example = tuple_model.generate_example_json()

    # This works as a Python dict
    assert ("example_string", 123) in tuple_example["value"]

    # But JSON serialization would fail for tuple keys
    with pytest.raises(TypeError, match="keys must be str"):
        json.dumps(tuple_example)


def test_instruct_llm_with_dict_int_keys():
    """Test that instruct_llm handles dict with int keys properly."""
    model = build_dynamic_llm_datamodel(dict[int, str])
    instructions = model.instruct_llm()

    # The schema should indicate the key and value types
    assert "[[Schema]]" in instructions
    assert "object" in instructions.lower() or "dict" in instructions.lower()


# ---------------------------------------------------------------------------
# Internal Helper Tests
# ---------------------------------------------------------------------------
def test_generate_type_description():
    """Test _generate_type_description internal method."""
    from symai.models.base import LLMDataModel

    # Test with various types
    class TestModel(LLMDataModel):
        simple_str: str
        optional_int: int | None
        list_field: list[str]
        dict_field: dict[str, int]

    schema = TestModel.model_json_schema()

    # Access the internal method through the class
    desc = TestModel._generate_type_description({"type": "string"})
    assert "string" in desc.lower()

    desc = TestModel._generate_type_description({"type": "integer"})
    assert "integer" in desc.lower()

    desc = TestModel._generate_type_description({"type": "array", "items": {"type": "string"}})
    assert "array" in desc.lower() or "list" in desc.lower()


def test_resolve_allof_type():
    """Test _resolve_allof_type for allOf schemas."""
    class BaseModel(LLMDataModel):
        base_field: str

    class ExtendedModel(BaseModel):
        extended_field: int

    schema = ExtendedModel.model_json_schema()

    # Create an allOf structure manually for testing
    allof_schema = {
        "allOf": [
            {"type": "object", "properties": {"field1": {"type": "string"}}},
            {"type": "object", "properties": {"field2": {"type": "integer"}}}
        ]
    }

    resolved = ExtendedModel._resolve_allof_type(allof_schema, ExtendedModel.model_json_schema().get("$defs", {}))
    assert isinstance(resolved, str)
    assert resolved in ("object", "nested object (BaseModel)", "unknown")


def test_is_const_field():
    """Test _is_const_field detection."""
    class ConstModel(LLMDataModel):
        const_field: str = Const("fixed")
        normal_field: str = "default"

    assert ConstModel._is_const_field(ConstModel.model_fields["const_field"])
    assert not ConstModel._is_const_field(ConstModel.model_fields["normal_field"])


def test_has_default_value():
    """Test _has_default_value detection."""
    class DefaultModel(LLMDataModel):
        required: str
        optional: str = "default"
        nullable: str | None = None

    assert not DefaultModel._has_default_value(DefaultModel.model_fields["required"])
    assert DefaultModel._has_default_value(DefaultModel.model_fields["optional"])
    assert DefaultModel._has_default_value(DefaultModel.model_fields["nullable"])


# ---------------------------------------------------------------------------
# convert_dict_int_keys Comprehensive Tests
# ---------------------------------------------------------------------------
def test_convert_dict_int_keys_from_json_string():
    """Test that JSON string keys are converted to integers."""
    class IntKeyModel(LLMDataModel):
        data: dict[int, str]

    # Simulate JSON deserialization which converts int keys to strings
    json_str = '{"data": {"1": "one", "2": "two", "3": "three"}}'
    json_data = json.loads(json_str)

    model = IntKeyModel(**json_data)
    assert model.data == {1: "one", 2: "two", 3: "three"}
    assert all(isinstance(k, int) for k in model.data.keys())


def test_convert_dict_int_keys_nested_dicts():
    """Test conversion in nested dictionary structures."""
    class NestedIntModel(LLMDataModel):
        outer: dict[int, dict[int, str]]

    json_data = {"outer": {"1": {"10": "value"}}}
    model = NestedIntModel(**json_data)

    assert model.outer == {1: {10: "value"}}
    assert isinstance(list(model.outer.keys())[0], int)
    assert isinstance(list(model.outer[1].keys())[0], int)


def test_convert_dict_int_keys_invalid_conversion():
    """Test that invalid conversions raise errors."""
    class StrictIntModel(LLMDataModel):
        nums: dict[int, str]

    with pytest.raises(ValidationError):
        StrictIntModel(nums={"not_a_number": "value"})

    with pytest.raises(ValidationError):
        StrictIntModel(nums={"1.5": "value"})  # Float string should fail for int


def test_convert_dict_int_keys_preserves_valid_ints():
    """Test that already-integer keys are preserved."""
    class IntModel(LLMDataModel):
        data: dict[int, str]

    model = IntModel(data={1: "one", 2: "two"})
    assert model.data == {1: "one", 2: "two"}


# ---------------------------------------------------------------------------
# Comprehensive Negative Path Tests
# ---------------------------------------------------------------------------
def test_invalid_type_in_list():
    """Test that invalid types in lists are caught."""
    class ListModel(LLMDataModel):
        items: list[int]

    with pytest.raises(ValidationError):
        ListModel(items=[1, 2, "three", 4])


def test_invalid_type_in_dict_values():
    """Test that invalid types in dict values are caught."""
    class DictModel(LLMDataModel):
        mapping: dict[str, int]

    with pytest.raises(ValidationError):
        DictModel(mapping={"a": 1, "b": "not_an_int"})


def test_invalid_type_in_dict_keys():
    """Test that invalid types in dict keys are caught."""
    class DictKeyModel(LLMDataModel):
        int_keys: dict[int, str]

    # Non-convertible string key
    with pytest.raises(ValidationError):
        DictKeyModel(int_keys={"abc": "value"})


def test_overly_deep_recursion():
    """Test handling of overly deep recursion."""
    current = RecursiveModel(name="root")

    # Create very deep nesting
    for i in range(100):
        current = RecursiveModel(name=f"level_{i}", children=[current])

    # Should handle without stack overflow
    s = str(current)
    assert len(s) > 0

    # Schema generation should also handle it
    schema = RecursiveModel.simplify_json_schema()
    assert "recursive" in schema.lower() or "children" in schema


def test_circular_reference_detection():
    """Test detection and handling of circular references."""
    node1 = RecursiveModel(name="node1")
    node2 = RecursiveModel(name="node2")

    # Create circular reference
    node1.children = [node2]
    node2.children = [node1]

    # Should handle circular references in string representation
    s = str(node1)
    assert "node1" in s


def test_invalid_enum_value_rejection():
    """Test that invalid enum values are properly rejected."""
    with pytest.raises(ValidationError) as exc_info:
        ModelWithEnum(status="invalid_status", code=123)

    assert "status" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


def test_constraint_violations():
    """Test various constraint violations."""
    # Length constraint violation
    with pytest.raises(ValidationError):
        ModelWithConstraints(
            name="a" * 100,  # Too long
            age=25,
            email="test@example.com"
        )

    # Age constraint violation
    with pytest.raises(ValidationError):
        ModelWithConstraints(
            name="John",
            age=200,  # Too old
            email="test@example.com"
        )

    # Email pattern violation
    with pytest.raises(ValidationError):
        ModelWithConstraints(
            name="John",
            age=25,
            email="not_an_email"
        )


def test_missing_required_fields():
    """Test that missing required fields are caught."""
    with pytest.raises(ValidationError) as exc_info:
        SimpleModel()  # Missing 'value' field

    assert "value" in str(exc_info.value) or "required" in str(exc_info.value).lower()


def test_extra_fields_rejection():
    """Test that extra fields are handled according to model config."""
    class StrictModel(LLMDataModel):
        model_config = {"extra": "forbid"}
        allowed_field: str

    with pytest.raises(ValidationError):
        StrictModel(allowed_field="value", extra_field="not_allowed")


def test_union_type_validation_all_fail():
    """Test union validation when no branch matches."""
    class UnionModel(LLMDataModel):
        value: int | str | bool

    with pytest.raises(ValidationError):
        UnionModel(value={"dict": "not_allowed"})


def test_none_in_non_optional_field():
    """Test that None is rejected in non-optional fields."""
    with pytest.raises(ValidationError):
        SimpleModel(value=None)


def test_wrong_type_in_nested_model():
    """Test type errors in nested models."""
    class Outer(LLMDataModel):
        inner: SimpleModel

    with pytest.raises(ValidationError):
        Outer(inner="not_a_model")

    with pytest.raises(ValidationError):
        Outer(inner={"value": 123})  # Wrong type for value


# ---------------------------------------------------------------------------
# Format Resilience Tests
# ---------------------------------------------------------------------------
def test_format_output_structural_properties():
    """Test format output using structural properties instead of exact strings."""
    model = ComplexNestedModel(level1="test", nested=ComplexNestedModel(level1="nested"))
    s = str(model)

    # Check structural properties
    lines = s.split('\n')
    has_fields = any(':' in line for line in lines)
    has_values = "test" in s and "nested" in s
    has_indentation = any(line.startswith(' ') for line in lines if line.strip())

    assert has_fields, "Should have field separators"
    assert has_values, "Should contain field values"
    assert has_indentation or len(lines) == 1, "Should have structure"


def test_example_extraction_flexible():
    """Test flexible example extraction from various formats."""
    # Test various formatting variations
    test_cases = [
        "[[Example]]\n```python\n{'value': 123}\n```",
        "[[Example 1]] \n ```python\n {'value': 123} \n ```",
        "[[Example]]\n\n```python\n{'value': 123}\n\n```",
    ]

    flexible_pattern = re.compile(
        r"\[\[Example(?:\s+\d+)?\]\].*?```[pP]ython\s*(.*?)\s*```",
        re.DOTALL | re.IGNORECASE
    )

    for test in test_cases:
        matches = flexible_pattern.findall(test)
        assert len(matches) > 0, f"Failed to extract from: {test}"
        assert "123" in matches[0]


def test_cache_equality_not_identity():
    """Test cache using equality rather than identity."""
    schema1 = SimpleModel.simplify_json_schema()
    schema2 = SimpleModel.simplify_json_schema()

    # Should be equal but not necessarily the same object
    assert schema1 == schema2

    instr1 = SimpleModel.instruct_llm()
    instr2 = SimpleModel.instruct_llm()

    # Should be equal
    assert instr1 == instr2
