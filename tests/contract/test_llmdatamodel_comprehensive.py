import json
import re
from typing import Any, Optional, Union
from enum import Enum

import pytest
from pydantic import Field, ValidationError

from symai.models.base import (
    LLMDataModel,
    build_dynamic_llm_datamodel,
    LengthConstraint,
    CustomConstraint,
    Const,
)


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
    result = model.format_field("empty_list", [], indent=0)
    assert result == "empty_list:\n"


def test_format_field_empty_dict():
    """Test formatting of empty dictionaries."""
    model = SimpleModel(value="test")
    result = model.format_field("empty_dict", {}, indent=0)
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
    result = model.format_field("nested", nested_data, indent=0)
    assert "level1:" in result
    assert "level2:" in result
    assert "level3:" in result


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
    result = model.format_field("mixed", mixed_list, indent=0)
    assert "- : 1" in result
    assert "- : string" in result
    assert "key: value" in result


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
    assert "value: test" in s


def test_str_with_enum_field():
    """Test string representation with enum fields."""
    model = ModelWithEnum(
        status=ModelWithEnum.Status.ACTIVE,
        code=200
    )
    s = str(model)
    assert "status: active" in s
    assert "code: 200" in s


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
    assert schema1 is schema2


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
    """Test that instruct_llm is properly cached."""
    instr1 = SimpleModel.instruct_llm()
    instr2 = SimpleModel.instruct_llm()
    assert instr1 is instr2


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
    result = model.format_field("deep", deep_nested, indent=0)
    assert "f: value" in result


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
