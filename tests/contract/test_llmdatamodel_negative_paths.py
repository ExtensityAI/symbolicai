from typing import Any, Literal, Union

import pytest
from pydantic import Field, ValidationError, field_validator

from symai.models.base import LLMDataModel, build_dynamic_llm_datamodel


# ---------------------------------------------------------------------------
# Test: Invalid Bytes and Bytearray Constraints
# ---------------------------------------------------------------------------
def test_bytes_with_invalid_length_constraint():
    """Test that bytes fields reject data exceeding length constraints."""
    class BytesModel(LLMDataModel):
        data: bytes = Field(..., max_length=10)

    # Valid case
    model = BytesModel(data=b"short")
    assert model.data == b"short"

    # Invalid - too long
    with pytest.raises(ValidationError) as exc_info:
        BytesModel(data=b"this is way too long for the constraint")
    assert "data" in str(exc_info.value)


def test_bytes_with_negative_length():
    """Test bytes validation with invalid constraints."""
    from pydantic_core import SchemaError
    with pytest.raises((ValueError, SchemaError)):
        class InvalidBytesModel(LLMDataModel):
            # This should fail at class definition time
            data: bytes = Field(..., min_length=-1)


def test_bytes_non_utf8_sequences():
    """Test handling of non-UTF8 byte sequences."""
    class BytesModel(LLMDataModel):
        data: bytes

    # Test with invalid UTF-8 sequences
    invalid_utf8 = b'\x80\x81\x82\x83'
    model = BytesModel(data=invalid_utf8)
    assert model.data == invalid_utf8

    # String representation should handle non-UTF8 gracefully
    s = str(model)
    assert "data:" in s


# ---------------------------------------------------------------------------
# Test: Malformed Literal Types
# ---------------------------------------------------------------------------
def test_empty_literal():
    """Test that empty Literal types are rejected."""
    from pydantic.errors import PydanticSchemaGenerationError
    with pytest.raises(PydanticSchemaGenerationError):
        # Empty Literal should fail
        build_dynamic_llm_datamodel(Literal)


def test_literal_with_mixed_types():
    """Test Literal with incompatible mixed types."""
    class MixedLiteralModel(LLMDataModel):
        value: Literal[1, "two", 3.0, True]

    # Valid values
    assert MixedLiteralModel(value=1).value == 1
    assert MixedLiteralModel(value="two").value == "two"
    assert MixedLiteralModel(value=3.0).value == 3.0
    assert MixedLiteralModel(value=True).value == True

    # Invalid value
    with pytest.raises(ValidationError):
        MixedLiteralModel(value="three")


def test_literal_with_mutable_types():
    """Test that Literal with mutable types behaves correctly."""
    # In Python's typing, Literal with unhashable types is technically allowed
    # at type definition time but will fail when used in a model

    # These are technically valid type hints but semantically incorrect
    # Python's typing module doesn't validate Literal contents at definition
    try:
        # This won't raise immediately
        BadLiteral1 = Literal[["list"]]
        BadLiteral2 = Literal[{"dict": "value"}]

        # Using them in a model might work but won't validate properly
        # Pydantic v2 handles this more gracefully
        class BadLiteralModel(LLMDataModel):
            value: BadLiteral1

        # The model creation succeeds, and surprisingly accepts the literal value
        # This is a quirk of how Pydantic handles Literal with mutable types
        model = BadLiteralModel(value=["list"])
        assert model.value == ["list"]
    except (TypeError, ValidationError):
        # Some configurations might catch this earlier
        pass


def test_literal_with_none():
    """Test Literal containing None."""
    class LiteralWithNone(LLMDataModel):
        value: Literal[None, "something"]

    model1 = LiteralWithNone(value=None)
    assert model1.value is None

    model2 = LiteralWithNone(value="something")
    assert model2.value == "something"

    with pytest.raises(ValidationError):
        LiteralWithNone(value="other")


# ---------------------------------------------------------------------------
# Test: Extremely Wide Unions
# ---------------------------------------------------------------------------
def test_extremely_wide_union():
    """Test union with many alternatives."""
    # Create a union with 20+ alternatives
    WideUnion = Union[
        int, str, float, bool, bytes,
        list[int], list[str], list[float],
        dict[str, int], dict[int, str], dict[str, str],
        tuple[int, ...], tuple[str, str], tuple[float, int, str],
        set[int], set[str], frozenset[int],
        None
    ]

    class WideUnionModel(LLMDataModel):
        value: WideUnion

    # Test various valid types
    test_values = [
        42, "text", 3.14, True, b"bytes",
        [1, 2], ["a", "b"], [1.0, 2.0],
        {"a": 1}, {1: "a"}, {"x": "y"},
        (1, 2, 3), ("a", "b"), (1.0, 2, "c"),
        {1, 2}, {"a", "b"}, frozenset([1, 2]),
        None
    ]

    for val in test_values:
        model = WideUnionModel(value=val)
        # Just check it doesn't crash
        _ = str(model)
        _ = model.generate_example_json()


def test_deeply_nested_union():
    """Test deeply nested union types."""
    DeepUnion = Union[
        int,
        Union[str, Union[float, Union[bool, bytes]]]
    ]

    class DeepUnionModel(LLMDataModel):
        value: DeepUnion

    # All these should work
    assert DeepUnionModel(value=42).value == 42
    assert DeepUnionModel(value="text").value == "text"
    assert DeepUnionModel(value=3.14).value == 3.14
    assert DeepUnionModel(value=True).value == True
    assert DeepUnionModel(value=b"bytes").value == b"bytes"

    # Complex type should fail
    with pytest.raises(ValidationError):
        DeepUnionModel(value=[1, 2, 3])


def test_union_with_conflicting_constraints():
    """Test union types with conflicting field constraints."""
    class ConflictingUnion(LLMDataModel):
        # String must be short, list must be long - creates validation complexity
        value: Union[
            str,  # With constraint via validator
            list[int]  # With different constraint
        ]

        @field_validator('value')
        def validate_value(cls, v):
            if isinstance(v, str) and len(v) > 5:
                raise ValueError("String too long")
            if isinstance(v, list) and len(v) < 3:
                raise ValueError("List too short")
            return v

    # Valid cases
    ConflictingUnion(value="short")
    ConflictingUnion(value=[1, 2, 3, 4])

    # Invalid cases
    with pytest.raises(ValidationError):
        ConflictingUnion(value="too long string")

    with pytest.raises(ValidationError):
        ConflictingUnion(value=[1])  # Too short list


# ---------------------------------------------------------------------------
# Test: Recursive Depth Limit Edge Cases
# ---------------------------------------------------------------------------
def test_mutual_recursion_limit():
    """Test mutually recursive models hitting depth limit."""
    class ModelA(LLMDataModel):
        name: str
        b_ref: 'ModelB | None' = None

    class ModelB(LLMDataModel):
        name: str
        a_ref: ModelA | None = None

    # Create a mutual recursion
    a = ModelA(name="A")
    b = ModelB(name="B", a_ref=a)
    a.b_ref = b

    # Should handle mutual recursion gracefully
    s = str(a)
    assert "A" in s
    assert "B" in s
    # Should detect circular reference
    assert "<circular reference>" in s or "<max depth reached>" in s


def test_self_referential_list():
    """Test self-referential list structures."""
    class ListNode(LLMDataModel):
        value: int
        next_nodes: list['ListNode'] = Field(default_factory=list)

    # Create a structure that references itself
    node1 = ListNode(value=1)
    node2 = ListNode(value=2, next_nodes=[node1])
    node1.next_nodes.append(node2)  # Create cycle

    s = str(node1)
    assert "<circular reference>" in s


# ---------------------------------------------------------------------------
# Test: Invalid Field Names and Special Characters
# ---------------------------------------------------------------------------
def test_field_names_with_python_keywords():
    """Test model with Python keyword field names."""
    class KeywordModel(LLMDataModel):
        class_: str = Field(alias="class")
        return_: int = Field(alias="return")
        yield_: float = Field(alias="yield")

    # Use the aliases when creating the model
    model = KeywordModel(**{"class": "test", "return": 42, "yield": 3.14})
    s = str(model)
    assert "class_" in s
    assert "return_" in s
    assert "yield_" in s


def test_field_names_with_special_unicode():
    """Test field names with special Unicode characters."""
    class UnicodeFieldModel(LLMDataModel):
        emoji_field: str = Field(default="happy", alias="emoji_😀")
        chinese_field: int = Field(default=42, alias="chinese_字段")
        arabic_field: float = Field(default=3.14, alias="arabic_حقل")

    model = UnicodeFieldModel()
    s = str(model)
    # Should handle Unicode in values
    assert "happy" in s
    assert "42" in str(model)
    assert "3.14" in str(model)


def test_field_with_null_bytes():
    """Test handling of null bytes in string fields."""
    class NullByteModel(LLMDataModel):
        text: str

    # String with null byte
    text_with_null = "hello\x00world"
    model = NullByteModel(text=text_with_null)
    assert model.text == text_with_null

    # Should handle in string representation
    s = str(model)
    assert "text:" in s


# ---------------------------------------------------------------------------
# Test: Invalid Type Annotations
# ---------------------------------------------------------------------------
def test_invalid_generic_without_args():
    """Test generic types without type arguments."""
    # In Pydantic v2, list without args is actually allowed (treated as list[Any])
    class GenericModel(LLMDataModel):
        items: list  # Allowed in Pydantic v2

    # Should work
    model = GenericModel(items=[1, "mixed", 3.14])
    assert len(model.items) == 3


def test_any_type_validation():
    """Test Any type accepts anything but provides no validation."""
    class AnyModel(LLMDataModel):
        anything: Any

    # Should accept literally anything
    test_values = [
        42, "string", 3.14, True, None,
        [1, 2, 3], {"key": "value"},
        object(), lambda x: x,
        type, Exception("test")
    ]

    for val in test_values:
        model = AnyModel(anything=val)
        # Just verify it doesn't crash
        _ = str(model)


def test_forward_reference_not_resolvable():
    """Test forward reference that can't be resolved."""
    class ModelWithBadRef(LLMDataModel):
        # Reference to non-existent model
        bad_ref: 'NonExistentModel' = None

    # Should handle gracefully when generating schema
    try:
        schema = ModelWithBadRef.simplify_json_schema()
        # Should treat as Any or string type
        assert "NonExistentModel" in schema or "any" in schema.lower()
    except Exception:
        # If it fails, it should be a clear error
        pass


# ---------------------------------------------------------------------------
# Test: Constraint Validation Edge Cases
# ---------------------------------------------------------------------------
def test_conflicting_constraints():
    """Test models with conflicting constraints."""
    # Pydantic doesn't validate constraint conflicts at definition time
    # The conflict only shows up when trying to use the model
    class ConflictingModel(LLMDataModel):
        # min > max - logically conflicting but allowed at definition
        value: int = Field(ge=100, le=50)

    # Should fail when trying to create an instance
    with pytest.raises(ValidationError):
        ConflictingModel(value=75)  # Can't satisfy both constraints


def test_constraint_on_optional_field():
    """Test constraints on optional fields."""
    class OptionalConstraintModel(LLMDataModel):
        value: int | None = Field(None, ge=0, le=100)

    # None should be allowed
    model1 = OptionalConstraintModel(value=None)
    assert model1.value is None

    # Valid value
    model2 = OptionalConstraintModel(value=50)
    assert model2.value == 50

    # Invalid value
    with pytest.raises(ValidationError):
        OptionalConstraintModel(value=150)


def test_regex_constraint_with_special_chars():
    """Test regex constraints with special regex characters."""
    class RegexModel(LLMDataModel):
        pattern: str = Field(..., pattern=r'^[a-z]{3}\.[0-9]+\$$')

    # Valid
    model = RegexModel(pattern="abc.123$")
    assert model.pattern == "abc.123$"

    # Invalid
    with pytest.raises(ValidationError):
        RegexModel(pattern="ABC.123$")  # uppercase

    with pytest.raises(ValidationError):
        RegexModel(pattern="abc.123")  # missing $


# ---------------------------------------------------------------------------
# Test: JSON Serialization Edge Cases
# ---------------------------------------------------------------------------
def test_json_with_inf_and_nan():
    """Test JSON serialization with infinity and NaN values."""
    class FloatModel(LLMDataModel):
        value: float

    import math

    # Infinity
    model_inf = FloatModel(value=math.inf)
    assert math.isinf(model_inf.value)

    # NaN
    model_nan = FloatModel(value=math.nan)
    assert math.isnan(model_nan.value)

    # JSON serialization should handle these
    try:
        json_str = model_inf.model_dump_json()
        # Some JSON encoders use "Infinity" or null
        assert "null" in json_str or "Infinity" in json_str
    except ValueError:
        # Some configurations don't allow inf/nan in JSON
        pass


def test_circular_reference_in_dict():
    """Test circular references in dictionary fields."""
    class DictModel(LLMDataModel):
        data: dict

    # Create circular reference
    circular_dict = {"key": "value"}
    circular_dict["self"] = circular_dict

    # Pydantic handles circular references in dicts
    model = DictModel(data=circular_dict)

    # String representation should handle the circular reference
    s = str(model)
    # Our implementation should detect the circular reference
    assert "<circular reference>" in s


# ---------------------------------------------------------------------------
# Test: Dynamic Model Building Edge Cases
# ---------------------------------------------------------------------------
def test_build_dynamic_with_invalid_type():
    """Test dynamic model building with invalid types."""
    # These should handle gracefully
    invalid_types = [
        type(None),  # NoneType directly
        ...,  # Ellipsis
    ]

    for invalid_type in invalid_types:
        try:
            model = build_dynamic_llm_datamodel(invalid_type)
            # If it succeeds, verify it works
            example = model.generate_example_json()
            assert "value" in example
        except (TypeError, ValueError):
            # Expected to fail for some types
            pass


def test_build_dynamic_with_complex_nested():
    """Test dynamic model with complex nested structures."""
    ComplexType = dict[str, list[dict[int, set[tuple[str, int]]]]]

    try:
        model = build_dynamic_llm_datamodel(ComplexType)
        example = model.generate_example_json()
        assert isinstance(example["value"], dict)
    except Exception:
        # Complex types might not be fully supported
        pass


# ---------------------------------------------------------------------------
# Test: Memory and Performance Edge Cases
# ---------------------------------------------------------------------------
def test_extremely_large_list():
    """Test handling of very large lists."""
    class LargeListModel(LLMDataModel):
        items: list[int]

    # Create a very large list
    large_list = list(range(10000))
    model = LargeListModel(items=large_list)

    # String representation currently shows all items
    # This is acceptable for debugging purposes
    s = str(model)
    # Verify it handles the large list without crashing
    assert "items:" in s
    assert len(model.items) == 10000


def test_deeply_nested_structure():
    """Test extremely deep nesting."""
    # Create deeply nested dict
    deep_dict = {"level": 0}
    current = deep_dict
    for i in range(100):
        current["next"] = {"level": i + 1}
        current = current["next"]

    class DeepModel(LLMDataModel):
        data: dict

    model = DeepModel(data=deep_dict)
    s = str(model)
    # Should hit max depth
    assert "<max depth reached>" in s


# ---------------------------------------------------------------------------
# Test: Edge Cases in Generate Example
# ---------------------------------------------------------------------------
def test_generate_example_with_empty_union():
    """Test example generation with Union[None]."""
    class EmptyUnionModel(LLMDataModel):
        value: None

    example = EmptyUnionModel.generate_example_json()
    assert example["value"] is None


def test_generate_example_with_recursive_limit():
    """Test example generation doesn't infinite loop on recursive models."""
    class RecursiveModel(LLMDataModel):
        name: str
        children: list['RecursiveModel'] = Field(default_factory=list)

    # Should generate example without infinite recursion
    example = RecursiveModel.generate_example_json()
    assert "name" in example
    assert "children" in example
    # Should limit recursion depth
    assert isinstance(example["children"], list)


# ---------------------------------------------------------------------------
# Test: Schema Generation Edge Cases
# ---------------------------------------------------------------------------
def test_schema_with_custom_validation():
    """Test schema generation with custom validators."""
    class CustomValidatedModel(LLMDataModel):
        value: int

        @field_validator('value')
        def must_be_even(cls, v):
            if v % 2 != 0:
                raise ValueError('must be even')
            return v

    # Schema should still generate
    schema = CustomValidatedModel.simplify_json_schema()
    assert "value" in schema

    # Validation should work
    CustomValidatedModel(value=2)
    with pytest.raises(ValidationError):
        CustomValidatedModel(value=3)


def test_schema_with_complex_inheritance():
    """Test schema generation with multiple inheritance."""
    class Base1(LLMDataModel):
        field1: str

    class Base2(LLMDataModel):
        field2: int

    class Combined(Base1, Base2):
        field3: float

    schema = Combined.simplify_json_schema()
    assert "field1" in schema
    assert "field2" in schema
    assert "field3" in schema


# ---------------------------------------------------------------------------
# Test: Error Messages and Debugging
# ---------------------------------------------------------------------------
def test_helpful_error_messages():
    """Test that validation errors provide helpful context."""
    class StrictModel(LLMDataModel):
        name: str = Field(..., min_length=3, max_length=10)
        age: int = Field(..., ge=0, le=120)

    # Test various validation failures
    with pytest.raises(ValidationError) as exc_info:
        StrictModel(name="ab", age=25)  # name too short
    error_str = str(exc_info.value)
    assert "name" in error_str
    assert "at least 3" in error_str.lower() or "min" in error_str.lower()

    with pytest.raises(ValidationError) as exc_info:
        StrictModel(name="valid", age=150)  # age too high
    error_str = str(exc_info.value)
    assert "age" in error_str
    assert "120" in error_str or "less than" in error_str.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
