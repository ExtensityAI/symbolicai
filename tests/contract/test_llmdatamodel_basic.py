import json
import re

import pytest
from pydantic import ValidationError

from symai.models.base import Const, LLMDataModel, build_dynamic_llm_datamodel


# ---------------------------------------------------------------------------
# Helper utilities used across multiple assertions
# ---------------------------------------------------------------------------
def extract_examples(text: str) -> list[str]:
    """Return all raw JSON strings embedded in ``[[Example]]`` blocks."""
    pattern = re.compile(r"\[\[Example(?: \d+)?]]\s+```json\s+(.*?)\s+```", re.DOTALL)
    return pattern.findall(text)


def parse_structure(text: str) -> dict:
    """Parse formatted text to extract structural properties instead of exact strings."""
    lines = text.split('\n')
    structure = {
        'has_header': any(line.strip().startswith('[[') and line.strip().endswith(']]') for line in lines),
        'has_list_items': any(line.strip().startswith('-') for line in lines),
        'field_names': [],
        'indented_sections': []
    }

    for line in lines:
        if ':' in line and not line.strip().startswith('-'):
            field_name = line.split(':')[0].strip()
            if field_name:
                structure['field_names'].append(field_name)
        if line.startswith('  ') and line.strip():
            structure['indented_sections'].append(line.strip())

    return structure


# ---------------------------------------------------------------------------
# Models used in different scenarios
# ---------------------------------------------------------------------------
class Address(LLMDataModel):
    street: str
    city: str


class Person(LLMDataModel):
    name: str
    age: int | None


class Thoughts(LLMDataModel):
    thoughts: list[str]


class User(LLMDataModel):
    name: str
    age: int | None
    address: Address
    tags: list[str]
    metadata: dict[str, int]


class ModelWithIntKeys(LLMDataModel):
    int_dict: dict[int, str]
    mixed_dict: dict[int, dict[str, int]]


# ---------------------------------------------------------------------------
# ``__str__`` and ``format_field`` behaviour
# ---------------------------------------------------------------------------
def test_str_representation_with_header_and_nested():
    """The string representation should include the optional header, nested models
    and the correct indentation structure.
    """
    user = User(
        name="Alice",
        age=30,
        address=Address(street="Main", city="NY"),
        tags=["engineer", "runner"],
        metadata={"score": 42},
        section_header="User Info",
    )

    s = str(user)
    structure = parse_structure(s)

    assert structure['has_header'], "Missing section header"
    assert s.startswith("[[User Info]]"), "Header should be at the beginning"

    assert "name" in structure['field_names']
    assert "age" in structure['field_names']
    assert "address" in structure['field_names']

    assert "Alice" in s
    assert "30" in s
    assert "Main" in s
    assert "NY" in s

    assert structure['has_list_items'], "List items should be formatted with bullet points"
    assert "score" in s


def test_str_includes_none_values():
    """Fields set to ``None`` should be included in the resulting string."""
    user = User(
        name="Bob",
        age=None,
        address=Address(street="Elm", city="LA"),
        tags=[],
        metadata={},
    )

    s = str(user)
    assert "None" in s, "Fields with a value of None should be rendered"


def test_str_with_invalid_nested_structure():
    """Test handling of invalid nested structures."""
    class BrokenModel(LLMDataModel):
        data: dict

    model = BrokenModel(data={"key": object()})
    s = str(model)
    assert "data" in s


# ---------------------------------------------------------------------------
# ``convert_dict_int_keys`` - Testing the pre-validator
# ---------------------------------------------------------------------------
def test_convert_dict_int_keys_from_json():
    """Test that JSON string keys are converted back to integers when appropriate."""
    json_data = {
        "int_dict": {"1": "value1", "2": "value2"},
        "mixed_dict": {"3": {"nested": 4}}
    }

    model = ModelWithIntKeys(**json_data)
    assert model.int_dict == {1: "value1", 2: "value2"}
    assert model.mixed_dict == {3: {"nested": 4}}
    assert isinstance(list(model.int_dict.keys())[0], int)


def test_convert_dict_int_keys_invalid():
    """Test that non-numeric string keys raise validation errors."""
    with pytest.raises(ValidationError):
        ModelWithIntKeys(
            int_dict={"not_a_number": "value"},
            mixed_dict={1: {"nested": 2}}
        )


# ---------------------------------------------------------------------------
# ``generate_example_json``
# ---------------------------------------------------------------------------
def test_generate_example_json_complex():
    """The helper should create sensible placeholder values for a variety of
    field types including primitives, containers, and nested models.
    """
    example = LLMDataModel.generate_example_json(User)

    assert isinstance(example["name"], str)
    assert isinstance(example["age"], int)

    address = example["address"]
    assert isinstance(address, dict)
    assert "street" in address and "city" in address

    assert isinstance(example["tags"], list)
    assert isinstance(example["metadata"], dict)


def test_generate_example_json_union_resolution():
    """For a simple ``int | str`` union the heuristic should choose one valid option."""
    DynModel = build_dynamic_llm_datamodel(int | str)
    example = LLMDataModel.generate_example_json(DynModel)

    assert "value" in example
    assert isinstance(example["value"], (int, str))


def test_generate_example_json_with_invalid_type():
    """Test handling of invalid or unsupported types."""
    class InvalidModel(LLMDataModel):
        model_config = {"arbitrary_types_allowed": True}
        custom_obj: object

    example = LLMDataModel.generate_example_json(InvalidModel)
    assert "custom_obj" in example


# ---------------------------------------------------------------------------
# ``simplify_json_schema``
# ---------------------------------------------------------------------------
def test_simplify_json_schema_contains_required_information():
    """A human-readable schema must contain field names and type descriptions."""
    schema_text = User.simplify_json_schema()

    assert schema_text.startswith("[[Schema]]")

    for field_name in ["name", "age", "address", "tags", "metadata"]:
        assert field_name in schema_text

    assert any(word in schema_text.lower() for word in ["string", "text", "str"])
    assert any(word in schema_text.lower() for word in ["integer", "number", "int"])
    assert any(word in schema_text.lower() for word in ["nested", "object"])


def test_simplify_json_schema_caching():
    """Schema generation should be cached for performance."""
    schema1 = User.simplify_json_schema()
    schema2 = User.simplify_json_schema()
    assert schema1 == schema2


def test_simplify_json_schema_with_const_fields():
    """Test schema generation with const fields."""
    class ConstModel(LLMDataModel):
        const_field: str = Const("fixed_value")
        normal_field: int

    schema = ConstModel.simplify_json_schema()
    assert "const_field" in schema
    assert "fixed_value" in schema


# ---------------------------------------------------------------------------
# ``instruct_llm``
# ---------------------------------------------------------------------------
def test_instruct_llm_single_example_for_standard_model():
    """Models without unions should yield exactly one example block."""
    instr = User.instruct_llm()

    assert "[[Result]]" in instr
    assert "[[Schema]]" in instr

    examples = extract_examples(instr)
    assert len(examples) == 1

    parsed = json.loads(examples[0])
    expected_keys = set(User.model_fields.keys()) - {"section_header"}
    assert set(parsed.keys()) == expected_keys


def test_instruct_llm_multiple_examples_for_union():
    """A union containing alternatives should lead to multiple example blocks."""
    DynModel = build_dynamic_llm_datamodel(list[Address] | dict[str, int])
    instr = DynModel.instruct_llm()

    examples = extract_examples(instr)
    assert len(examples) >= 2, "Expected multiple examples for union types"

    types_found = set()
    for example_str in examples:
        parsed = json.loads(example_str)
        types_found.add(type(parsed["value"]))

    assert list in types_found and dict in types_found


def test_instruct_llm_caching():
    """Instruction generation should be cached."""
    instr1 = User.instruct_llm()
    instr2 = User.instruct_llm()
    assert instr1 == instr2


def test_union_int_str():
    """Union of two primitive types should generate appropriate examples."""
    model = build_dynamic_llm_datamodel(int | str)
    instr = model.instruct_llm()

    assert any(word in instr for word in ["integer", "string", "int", "str"])

    ex_blocks = extract_examples(instr)
    assert len(ex_blocks) >= 1

    value_types = set()
    for block in ex_blocks:
        parsed = json.loads(block)
        value_types.add(type(parsed["value"]))

    assert any(t in value_types for t in [int, str])


def test_union_with_optional():
    """`str | None` should handle the None case appropriately."""
    model = build_dynamic_llm_datamodel(str | None)
    instr = model.instruct_llm()
    ex_blocks = extract_examples(instr)

    assert len(ex_blocks) >= 1

    for block in ex_blocks:
        parsed = json.loads(block)
        assert parsed["value"] is None or isinstance(parsed["value"], str)


def test_user_defined_model_non_union():
    """Ensure instruct_llm works for normal user-defined models."""
    instr = Thoughts.instruct_llm()
    examples = extract_examples(instr)
    assert len(examples) >= 1

    parsed = json.loads(examples[0])
    assert "thoughts" in parsed
    assert isinstance(parsed["thoughts"], list)


def test_nested_union_complex():
    """Union with complex nested types."""
    union_t = list[Person] | dict[str, list[int]]
    model = build_dynamic_llm_datamodel(union_t)
    instr = model.instruct_llm()

    ex_blocks = extract_examples(instr)
    assert len(ex_blocks) >= 1

    for block in ex_blocks:
        parsed = json.loads(block)
        value = parsed["value"]
        assert isinstance(value, (list, dict))


# ---------------------------------------------------------------------------
# Default ``validate`` and ``remedy``
# ---------------------------------------------------------------------------
def test_default_validate_and_remedy_are_noops():
    """The default implementation should return ``None``."""
    m = Address(street="A", city="B")
    assert m.validate() is None
    assert m.remedy() is None


# ---------------------------------------------------------------------------
# Negative Path Tests
# ---------------------------------------------------------------------------
def test_invalid_field_types():
    """Test that invalid field types raise appropriate errors."""
    with pytest.raises(ValidationError):
        User(
            name=123,
            age="not_an_int",
            address="not_an_address",
            tags="not_a_list",
            metadata=[1, 2, 3]
        )


def test_missing_required_fields():
    """Test that missing required fields raise validation errors."""
    with pytest.raises(ValidationError):
        User(age=25)


def test_const_field_validation():
    """Test that const fields enforce their values."""
    class StrictModel(LLMDataModel):
        const_val: str = Const("must_be_this")

    model = StrictModel()
    assert model.const_val == "must_be_this"

    with pytest.raises(ValidationError):
        StrictModel(const_val="something_else")


def test_invalid_enum_value():
    """Test that invalid enum values are rejected."""
    from enum import Enum

    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    class StatusModel(LLMDataModel):
        status: Status

    with pytest.raises(ValidationError):
        StatusModel(status="invalid_status")


def test_deeply_nested_validation_error():
    """Test that validation errors in deeply nested structures are properly reported."""
    class Level3(LLMDataModel):
        value: int

    class Level2(LLMDataModel):
        level3: Level3

    class Level1(LLMDataModel):
        level2: Level2

    with pytest.raises(ValidationError):
        Level1(level2={"level3": {"value": "not_an_int"}})


def test_union_validation_all_branches_fail():
    """Test that union validation fails when no branch matches."""
    DynModel = build_dynamic_llm_datamodel(int | str)

    with pytest.raises(ValidationError):
        DynModel(value=[1, 2, 3])


def test_recursive_model_depth_limit():
    """Test handling of recursive models with extreme depth."""
    class RecursiveModel(LLMDataModel):
        value: str
        child: 'RecursiveModel | None' = None

    current = None
    for i in range(1000):
        current = RecursiveModel(value=f"level_{i}", child=current)

    s = str(current)
    assert len(s) > 0
    assert "level_" in s


def test_malformed_json_schema():
    """Test handling of models that might produce malformed schemas."""
    class WeirdModel(LLMDataModel):
        field_with_very_long_name_that_might_cause_issues_in_formatting: str
        field_with_special_chars_αβγ: int

    schema = WeirdModel.simplify_json_schema()
    assert "field_with_very_long_name" in schema

    example = LLMDataModel.generate_example_json(WeirdModel)
    assert isinstance(example, dict)


def test_circular_reference_in_union():
    """Test handling of circular references in union types."""
    class Node(LLMDataModel):
        value: str
        children: list['Node'] | None = None

    node = Node(value="root", children=[
        Node(value="child1"),
        Node(value="child2", children=[Node(value="grandchild")])
    ])

    s = str(node)
    assert "root" in s
    assert "child1" in s

    schema = Node.simplify_json_schema()
    assert "recursive" in schema.lower() or "children" in schema
