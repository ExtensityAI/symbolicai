import json
import re
from typing import Any, Optional, Union, Literal
from enum import Enum
from unittest.mock import patch

import pytest
from pydantic import Field, ValidationError, field_validator

from symai.models.base import (
    LLMDataModel,
    build_dynamic_llm_datamodel,
    Const,
)


# ---------------------------------------------------------------------------
# Advanced Test Models
# ---------------------------------------------------------------------------
class ModelWithValidation(LLMDataModel):
    email: str
    age: int

    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

    @field_validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Invalid age')
        return v

    def validate(self) -> Optional[str]:
        if self.age < 18:
            return "User must be 18 or older"
        return None

    def remedy(self) -> Optional[str]:
        if self.age < 18:
            return "Set age to 18 for minimum requirement"
        return None


class ModelWithLiterals(LLMDataModel):
    mode: Literal["read", "write", "execute"]
    level: Literal[1, 2, 3]
    flag: Literal[True]


class ModelWithMultipleUnions(LLMDataModel):
    simple_union: Union[int, str]
    complex_union: Union[list[int], dict[str, str], None]
    nested_union: Union[list[Union[int, str]], dict[str, Union[bool, float]]]


class ModelWithAllFieldTypes(LLMDataModel):
    # Primitives
    str_field: str
    int_field: int
    float_field: float
    bool_field: bool

    # Optional primitives
    opt_str: Optional[str]
    opt_int: Optional[int]

    # Collections
    list_field: list[str]
    dict_field: dict[str, int]
    set_field: set[int]
    tuple_field: tuple[str, int, bool]

    # Nested
    nested_model: 'SimpleNestedModel'
    list_of_models: list['SimpleNestedModel']
    dict_of_models: dict[str, 'SimpleNestedModel']


class SimpleNestedModel(LLMDataModel):
    name: str
    value: int


class ModelWithBytes(LLMDataModel):
    model_config = {"arbitrary_types_allowed": True}
    binary_data: bytes
    bytearray_data: bytearray


class ModelWithCustomTypes(LLMDataModel):
    class InnerEnum(str, Enum):
        OPTION_A = "a"
        OPTION_B = "b"

    enum_field: InnerEnum
    literal_union: Union[Literal["x"], Literal["y"], Literal["z"]]


# ---------------------------------------------------------------------------
# validate() and remedy() Tests
# ---------------------------------------------------------------------------
def test_validate_returns_error_message():
    """Test that validate returns appropriate error messages."""
    model = ModelWithValidation(email="test@example.com", age=16)
    error = model.validate()
    assert error == "User must be 18 or older"


def test_validate_returns_none_when_valid():
    """Test that validate returns None for valid data."""
    model = ModelWithValidation(email="test@example.com", age=25)
    assert model.validate() is None


def test_remedy_provides_fix_suggestion():
    """Test that remedy provides fix suggestions."""
    model = ModelWithValidation(email="test@example.com", age=16)
    suggestion = model.remedy()
    assert suggestion == "Set age to 18 for minimum requirement"


def test_remedy_returns_none_when_no_fix_needed():
    """Test that remedy returns None when no fix is needed."""
    model = ModelWithValidation(email="test@example.com", age=25)
    assert model.remedy() is None


# ---------------------------------------------------------------------------
# Literal Types Tests
# ---------------------------------------------------------------------------
def test_literal_types_in_schema():
    """Test that literal types are properly represented in schema."""
    schema = ModelWithLiterals.simplify_json_schema()
    assert "mode" in schema
    assert "level" in schema
    assert "flag" in schema
    assert any(x in schema for x in ["read", "write", "execute"])


def test_literal_types_in_example():
    """Test that literal types generate correct examples."""
    example = LLMDataModel.generate_example_json(ModelWithLiterals)
    assert example["mode"] in ["read", "write", "execute"]
    assert example["level"] in [1, 2, 3]
    assert example["flag"] is True


def test_literal_union_handling():
    """Test handling of unions of literal types."""
    example = LLMDataModel.generate_example_json(ModelWithCustomTypes)
    assert example["literal_union"] in ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Complex Union Resolution Tests
# ---------------------------------------------------------------------------
def test_union_resolution_with_none():
    """Test that None is handled correctly in unions."""
    DynModel = build_dynamic_llm_datamodel(Union[str, None])
    example = LLMDataModel.generate_example_json(DynModel)
    assert example["value"] == "example_string"


def test_union_resolution_nested_containers():
    """Test union resolution with nested container types."""
    DynModel = build_dynamic_llm_datamodel(
        Union[list[list[int]], dict[str, dict[str, str]]]
    )
    instr = DynModel.instruct_llm()
    pattern = re.compile(r"\[\[Example(?: \d+)?]]\s+```json\s+(.*?)\s+```", re.DOTALL)
    examples = pattern.findall(instr)
    assert len(examples) == 2


def test_union_with_basemodel():
    """Test union containing BaseModel subclasses."""
    DynModel = build_dynamic_llm_datamodel(
        Union[SimpleNestedModel, dict[str, Any]]
    )
    example = LLMDataModel.generate_example_json(DynModel)
    assert "value" in example


# ---------------------------------------------------------------------------
# format_field Advanced Tests
# ---------------------------------------------------------------------------
def test_format_field_with_bytes():
    """Test formatting of bytes and bytearray fields."""
    model = ModelWithBytes(
        binary_data=b"hello",
        bytearray_data=bytearray(b"world")
    )
    s = str(model)
    assert "binary_data:" in s
    assert "bytearray_data:" in s


def test_format_field_with_set():
    """Test formatting of set fields."""
    model = ModelWithAllFieldTypes(
        str_field="test",
        int_field=42,
        float_field=3.14,
        bool_field=True,
        opt_str="optional",
        opt_int=None,
        list_field=["a", "b"],
        dict_field={"key": 1},
        set_field={1, 2, 3},
        tuple_field=("x", 1, False),
        nested_model=SimpleNestedModel(name="nested", value=100),
        list_of_models=[SimpleNestedModel(name="item", value=1)],
        dict_of_models={"key": SimpleNestedModel(name="dict_item", value=2)}
    )
    s = str(model)
    assert "set_field:" in s
    assert "tuple_field:" in s


def test_format_field_with_large_indent():
    """Test formatting with large indentation values."""
    model = SimpleNestedModel(name="test", value=42)
    formatted = model.format_field("key", "value", indent=100)
    assert formatted.startswith(" " * 100)


def test_format_field_empty_string():
    """Test formatting empty string values."""
    model = SimpleNestedModel(name="", value=0)
    s = str(model)
    assert "name: " in s
    assert "value: 0" in s


# ---------------------------------------------------------------------------
# Schema Generation Edge Cases
# ---------------------------------------------------------------------------
def test_schema_with_forward_references():
    """Test schema generation with forward references."""
    class ForwardRefModel(LLMDataModel):
        self_ref: Optional['ForwardRefModel'] = None
        list_ref: list['ForwardRefModel'] = Field(default_factory=list)

    schema = ForwardRefModel.simplify_json_schema()
    assert "self_ref" in schema
    assert "list_ref" in schema


def test_schema_with_complex_defaults():
    """Test schema with complex default values."""
    class ComplexDefaultModel(LLMDataModel):
        list_default: list[int] = Field(default_factory=lambda: [1, 2, 3])
        dict_default: dict[str, str] = Field(default_factory=lambda: {"a": "b"})

    example = LLMDataModel.generate_example_json(ComplexDefaultModel)
    assert example["list_default"] == [1, 2, 3]
    assert example["dict_default"] == {"a": "b"}


def test_schema_caching_across_instances():
    """Test that schema caching works across different instances."""
    model1 = SimpleNestedModel(name="test1", value=1)
    model2 = SimpleNestedModel(name="test2", value=2)

    schema1 = model1.simplify_json_schema()
    schema2 = model2.simplify_json_schema()
    assert schema1 is schema2


# ---------------------------------------------------------------------------
# instruct_llm Advanced Tests
# ---------------------------------------------------------------------------
def test_instruct_llm_with_all_field_types():
    """Test instruction generation with all supported field types."""
    instr = ModelWithAllFieldTypes.instruct_llm()
    assert "str_field" in instr
    assert "int_field" in instr
    assert "float_field" in instr
    assert "bool_field" in instr
    assert "set_field" in instr
    assert "tuple_field" in instr


def test_instruct_llm_union_example_generation():
    """Test that union types generate appropriate number of examples."""
    instr = ModelWithMultipleUnions.instruct_llm()
    pattern = re.compile(r"\[\[Example(?: \d+)?]]\s+```json\s+(.*?)\s+```", re.DOTALL)
    examples = pattern.findall(instr)

    for example_str in examples:
        parsed = json.loads(example_str)
        assert "simple_union" in parsed
        assert "complex_union" in parsed
        assert "nested_union" in parsed


def test_instruct_llm_with_empty_model():
    """Test instruction generation for model with no fields."""
    class EmptyModel(LLMDataModel):
        pass

    instr = EmptyModel.instruct_llm()
    assert "[[Result]]" in instr
    assert "[[Schema]]" in instr


# ---------------------------------------------------------------------------
# Dynamic Model Building Advanced Tests
# ---------------------------------------------------------------------------
def test_build_dynamic_with_any_type():
    """Test dynamic model with Any type."""
    DynModel = build_dynamic_llm_datamodel(Any)
    instance = DynModel(value="anything")
    assert instance.value == "anything"
    instance2 = DynModel(value={"complex": ["structure"]})
    assert instance2.value == {"complex": ["structure"]}


def test_build_dynamic_with_union_of_unions():
    """Test dynamic model with nested unions."""
    DynModel = build_dynamic_llm_datamodel(
        Union[Union[int, str], Union[list, dict]]
    )
    instance1 = DynModel(value=42)
    instance2 = DynModel(value="text")
    instance3 = DynModel(value=[1, 2, 3])
    instance4 = DynModel(value={"key": "value"})

    assert instance1.value == 42
    assert instance2.value == "text"
    assert instance3.value == [1, 2, 3]
    assert instance4.value == {"key": "value"}


def test_build_dynamic_field_description():
    """Test that dynamic models have proper field descriptions."""
    DynModel = build_dynamic_llm_datamodel(str)
    field_info = DynModel.model_fields["value"]
    assert field_info.description is not None
    assert "dynamically generated" in field_info.description


# ---------------------------------------------------------------------------
# Error Handling and Validation Tests
# ---------------------------------------------------------------------------
def test_validation_error_on_invalid_data():
    """Test that validation errors are raised for invalid data."""
    with pytest.raises(ValidationError):
        ModelWithValidation(email="invalid", age=200)


def test_validation_with_pydantic_validators():
    """Test that Pydantic validators work correctly."""
    with pytest.raises(ValidationError) as exc_info:
        ModelWithValidation(email="no-at-sign", age=30)

    assert "Invalid email" in str(exc_info.value)


def test_const_field_immutability():
    """Test that const fields cannot be modified after creation."""
    class ConstModel(LLMDataModel):
        const_value: str = Const("fixed")

    model = ConstModel()
    assert model.const_value == "fixed"

    with pytest.raises(ValidationError):
        ConstModel(const_value="different")


# ---------------------------------------------------------------------------
# Performance and Memory Tests
# ---------------------------------------------------------------------------
def test_large_model_serialization():
    """Test handling of large models with many fields."""
    from pydantic import create_model

    # Create a model with 100 fields using create_model
    fields = {f'field_{i}': (str, ...) for i in range(100)}
    LargeModel = create_model('LargeModel', __base__=LLMDataModel, **fields)

    schema = LargeModel.simplify_json_schema()
    assert "field_0" in schema
    assert "field_99" in schema


def test_deep_recursion_limit():
    """Test that deep recursion doesn't cause stack overflow."""
    class DeepModel(LLMDataModel):
        value: str
        nested: Optional['DeepModel'] = None

    current = DeepModel(value="leaf")
    for i in range(100):
        current = DeepModel(value=f"level_{i}", nested=current)

    s = str(current)
    assert "level_0" in s
    assert "leaf" in s


# ---------------------------------------------------------------------------
# Special Character and Encoding Tests
# ---------------------------------------------------------------------------
def test_json_injection_prevention():
    """Test that JSON injection attempts are handled safely."""
    class InjectionModel(LLMDataModel):
        data: str

    model = InjectionModel(data='"},"injected":"value","original":"')
    example = LLMDataModel.generate_example_json(InjectionModel)
    json_str = json.dumps(example)
    parsed = json.loads(json_str)
    assert "injected" not in parsed


def test_null_byte_handling():
    """Test handling of null bytes in strings."""
    class NullByteModel(LLMDataModel):
        text: str

    model = NullByteModel(text="before\x00after")
    s = str(model)
    assert "before\x00after" in s or "before" in s


def test_unicode_normalization():
    """Test handling of different unicode normalizations."""
    class UnicodeModel(LLMDataModel):
        text: str

    model = UnicodeModel(text="café")  # é can be one or two unicode chars
    s = str(model)
    assert "caf" in s
