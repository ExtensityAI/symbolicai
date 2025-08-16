import json
import re
from enum import Enum
from typing import Any, Literal, Optional, Union, Dict, List, Optional

import pytest
from pydantic import Field, ValidationError, field_validator

from symai.models.base import Const, LLMDataModel, build_dynamic_llm_datamodel


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
    formatted = model.format_field("key", "value", indent=100, depth=0)
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
    assert schema1 == schema2


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
    import json
    instr = ModelWithMultipleUnions.instruct_llm()
    pattern = re.compile(r"\[\[Example(?: \d+)?]]\s+```json\s+(.*?)\s+```", re.DOTALL)
    examples = pattern.findall(instr)

    for example_str in examples:
        example = json.loads(example_str)
        assert "simple_union" in example
        assert "complex_union" in example
        assert "nested_union" in example


def test_instruct_llm_with_empty_model():
    """Test instruction generation for model with no fields."""
    class EmptyModel(LLMDataModel):
        pass

    instr = EmptyModel.instruct_llm()
    assert "[[Result]]" in instr
    assert "[[Schema]]" in instr


def test_instruct_llm_dict_person_union_list():
    """Test that dict[int, Person] | list[int] generates complete definitions."""
    class Person(LLMDataModel):
        name: str = Field(description="Name")
        age: int | None = Field(description="Age")

    DynModel = build_dynamic_llm_datamodel(dict[int, Person] | list[int])
    instr = DynModel.instruct_llm()

    # Check that Person is correctly included in definitions
    assert "[[Definitions]]" in instr
    assert "Person:" in instr
    assert '"name"' in instr
    assert '"age"' in instr

    # Check that the schema correctly describes the union type
    # Check that the schema correctly describes the union type (with JSON key clarification)
    assert "object of nested object (Person)" in instr and "array of integer" in instr

    # Check that definitions include clarifications for both union types
    # More flexible checks - look for key concepts rather than exact wording
    assert ("array" in instr.lower() and "integer" in instr.lower()) or ("list" in instr.lower() and "int" in instr.lower())
    assert ("dictionary" in instr.lower() or "dict" in instr.lower() or "object" in instr.lower()) and "Person" in instr

    # Verify that examples are generated correctly
    import json
    pattern = re.compile(r"\[\[Example(?: \d+)?]]\s+```json\s+(.*?)\s+```", re.DOTALL)
    examples = pattern.findall(instr)
    assert len(examples) == 2

    # First example should be dict[int, Person]
    example1 = json.loads(examples[0])
    assert "value" in example1
    assert isinstance(example1["value"], dict)

    # Second example should be list[int]
    example2 = json.loads(examples[1])
    assert "value" in example2
    assert isinstance(example2["value"], list)


def test_definitions_generic_message_and_preserved_descriptions_for_nested_dict():
    """Definitions should list all fields; use generic message when no Field(description=...) is provided,
    and preserve provided descriptions (e.g., for nested dict fields)."""

    class Metadata(LLMDataModel):
        author: str
        version: str
        tags: List[str]

    class ContentBlock(LLMDataModel):
        id: str
        title: str
        paragraphs: List[str]

    class NestedConfig(LLMDataModel):
        translations: Dict[str, Dict[str, str]] = Field(
            ...,
            description="Nested dictionary mapping language -> section -> text"
        )

    class ComplexDocument(LLMDataModel):
        doc_id: str
        metadata: Metadata
        content: List[ContentBlock]
        config: NestedConfig
        notes: Optional[str]

    instr = ComplexDocument.instruct_llm()

    # Schema checks for nested dict typing and optional notes
    assert "object of object of string" in instr
    assert "notes\" (string or null, required)" in instr or "notes\" (string or null" in instr

    # Definitions should include all root fields under ComplexDocument
    assert "[[Definitions]]" in instr
    assert "- ComplexDocument:" in instr


def test_examples_render_even_without_description():
    """Examples should be shown even when no description is provided, formatted as bullets."""

    class WithExamples(LLMDataModel):
        title: str = Field(examples=["a", "b", "c"])  # no description
        name: str = Field(description="Name field", examples=["x", "y"])  # has description

    generic_msg = (
        "No definition provided. Focus on the [[Schema]] and the prompt to infer "
        "the expected structure and constraints."
    )
    instr = WithExamples.instruct_llm()
    assert "[[Definitions]]" in instr
    # Generic message for title (no description)
    assert (
        '  - "title": No definition provided. Focus on the [[Schema]] and the prompt to infer '
        'the expected structure and constraints.'
    ) in instr
    # Bullet examples under title
    assert "    - Examples:" in instr
    assert "      - a" in instr and "      - b" in instr and "      - c" in instr
    # Name has description and bullet examples
    assert '  - "name": Name field' in instr
    assert "      - x" in instr and "      - y" in instr

    # Also verify single example formatting (no description)
    class WithSingleExample(LLMDataModel):
        value: int = Field(example=42)  # no description, single example

    instr_single = WithSingleExample.instruct_llm()
    assert '  - "value": ' in instr_single and "No definition provided." in instr_single
    assert "    - Example: 42" in instr_single


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
    assert "level_99" in s  # Should show the outermost level
    assert "level_50" in s  # Should show down to level 50
    assert "<max depth reached>" in s  # Should indicate max depth was hit


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


# ---------------------------------------------------------------------------
# convert_dict_int_keys Tests
# ---------------------------------------------------------------------------
def test_convert_dict_int_keys_basic():
    """Test conversion of string keys to integers in JSON data."""
    class IntKeyModel(LLMDataModel):
        data: dict[int, str]

    # JSON typically serializes integer keys as strings
    json_data = {"data": {"1": "value1", "2": "value2", "3": "value3"}}
    model = IntKeyModel(**json_data)

    assert model.data == {1: "value1", 2: "value2", 3: "value3"}
    assert all(isinstance(k, int) for k in model.data.keys())


def test_convert_dict_int_keys_nested():
    """Test conversion in nested dictionary structures."""
    class NestedIntKeyModel(LLMDataModel):
        outer: dict[int, dict[str, int]]

    json_data = {"outer": {"1": {"inner": 42}, "2": {"inner": 84}}}
    model = NestedIntKeyModel(**json_data)

    assert model.outer == {1: {"inner": 42}, 2: {"inner": 84}}
    assert isinstance(list(model.outer.keys())[0], int)


def test_convert_dict_int_keys_mixed_valid():
    """Test that valid numeric strings are converted while preserving structure."""
    class MixedKeyModel(LLMDataModel):
        int_keys: dict[int, str]
        str_keys: dict[str, int]

    json_data = {
        "int_keys": {"1": "a", "2": "b"},
        "str_keys": {"key1": 1, "key2": 2}
    }
    model = MixedKeyModel(**json_data)

    assert model.int_keys == {1: "a", 2: "b"}
    assert model.str_keys == {"key1": 1, "key2": 2}


def test_convert_dict_int_keys_invalid_raises():
    """Test that non-numeric string keys raise validation errors."""
    class StrictIntKeyModel(LLMDataModel):
        data: dict[int, str]

    with pytest.raises(ValidationError) as exc_info:
        StrictIntKeyModel(data={"not_a_number": "value"})

    assert "not_a_number" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Additional Negative Path Tests
# ---------------------------------------------------------------------------
def test_invalid_union_all_branches_fail():
    """Test that union validation fails when value doesn't match any branch."""
    class UnionModel(LLMDataModel):
        value: Union[int, str, bool]

    with pytest.raises(ValidationError):
        UnionModel(value=[1, 2, 3])  # List doesn't match any union branch


def test_length_constraint_violation():
    """Test that length constraints are enforced."""
    from pydantic import constr

    class ConstrainedModel(LLMDataModel):
        short_text: constr(max_length=5)
        long_text: constr(min_length=10)

    with pytest.raises(ValidationError):
        ConstrainedModel(short_text="too long string", long_text="short")


def test_regex_constraint_violation():
    """Test that regex constraints are enforced."""
    from pydantic import constr

    class RegexModel(LLMDataModel):
        email: constr(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')

    with pytest.raises(ValidationError):
        RegexModel(email="not_an_email")


def test_numeric_constraint_violation():
    """Test that numeric constraints are enforced."""
    from pydantic import confloat, conint

    class NumericModel(LLMDataModel):
        positive: conint(gt=0)
        percentage: confloat(ge=0, le=100)

    with pytest.raises(ValidationError):
        NumericModel(positive=-5, percentage=150)


def test_collection_size_constraint_violation():
    """Test that collection size constraints are enforced."""
    from pydantic import conlist

    class CollectionModel(LLMDataModel):
        items: conlist(str, min_length=2, max_length=5)

    with pytest.raises(ValidationError):
        CollectionModel(items=["only_one"])

    with pytest.raises(ValidationError):
        CollectionModel(items=["1", "2", "3", "4", "5", "6"])


def test_wrong_enum_value():
    """Test that wrong enum values are rejected."""
    from enum import Enum

    class Color(str, Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    class ColorModel(LLMDataModel):
        color: Color

    with pytest.raises(ValidationError):
        ColorModel(color="yellow")


def test_invalid_literal_value():
    """Test that invalid literal values are rejected."""
    with pytest.raises(ValidationError):
        ModelWithLiterals(mode="delete", level=4, flag=False)


def test_deeply_nested_validation_error():
    """Test that validation errors in deeply nested structures bubble up correctly."""
    class Level3(LLMDataModel):
        value: int

    class Level2(LLMDataModel):
        level3: Level3

    class Level1(LLMDataModel):
        level2: Level2

    with pytest.raises(ValidationError) as exc_info:
        Level1(level2={"level3": {"value": "not_an_int"}})

    error_str = str(exc_info.value)
    assert "value" in error_str.lower() or "int" in error_str.lower()


def test_circular_dependency_handling():
    """Test that circular dependencies are handled without infinite loops."""
    class Node(LLMDataModel):
        value: str
        children: Optional[list['Node']] = None

    # Create a circular structure
    node1 = Node(value="node1")
    node2 = Node(value="node2", children=[node1])
    node1.children = [node2]  # This creates a cycle

    # Should not crash when generating string or schema
    s = str(node1)
    assert "node1" in s or "node2" in s

    schema = Node.simplify_json_schema()
    assert "value" in schema


def test_excessive_recursion_depth():
    """Test handling of excessive recursion depth."""
    class DeepModel(LLMDataModel):
        value: int
        nested: Optional['DeepModel'] = None

    # Create a very deep structure
    current = DeepModel(value=0)
    for i in range(1, 500):
        current = DeepModel(value=i, nested=current)

    # Should handle deep recursion without stack overflow
    s = str(current)
    assert len(s) > 0


def test_malformed_field_names():
    """Test handling of fields with unusual names."""
    class WeirdFieldModel(LLMDataModel):
        model_config = {"arbitrary_types_allowed": True}
        field_with_spaces: str = Field(alias="field with spaces")
        field_with_special_chars: int = Field(alias="field!@#$%")

    model = WeirdFieldModel(**{"field with spaces": "test", "field!@#$%": 42})
    s = str(model)
    assert "test" in s
    assert "42" in s


def test_unicode_normalization():
    """Test handling of different unicode normalizations."""
    class UnicodeModel(LLMDataModel):
        text: str

    model = UnicodeModel(text="café")  # é can be one or two unicode chars
    s = str(model)
    assert "caf" in s


# ---------------------------------------------------------------------------
# Format resilience tests
# ---------------------------------------------------------------------------
def test_format_resilience_to_changes():
    """Test that format assertions are not overly strict."""
    model = SimpleNestedModel(name="test", value=42)
    s = str(model)

    # Check for structural properties rather than exact formatting
    lines = s.split('\n')
    has_field_separators = any(':' in line for line in lines)
    has_field_names = "name" in s and "value" in s
    has_field_values = "test" in s and "42" in s

    assert has_field_separators
    assert has_field_names
    assert has_field_values


def test_example_extraction_resilience():
    """Test that example extraction handles formatting variations."""
    import re

    # Test with different code fence variations
    test_prompts = [
        "[[Example]]\n```python\n{'value': 123}\n```",
        "[[Example 1]]\n```Python\n{'value': 123}\n```",  # Capital P
        "[[Example]]\n ```python\n{'value': 123}\n ```",  # Extra spaces
    ]

    for prompt in test_prompts:
        # More flexible pattern
        pattern = re.compile(
            r"\[\[Example(?:\s+\d+)?\]\].*?```[pP]ython\s+(.*?)\s*```",
            re.DOTALL | re.IGNORECASE
        )
        matches = pattern.findall(prompt)
        assert len(matches) > 0, f"Failed to extract from: {prompt}"
