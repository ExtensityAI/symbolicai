import json
import re

import pytest

from symai.models.base import (
    LLMDataModel,
    build_dynamic_llm_datamodel,
)


# ---------------------------------------------------------------------------
# Helper utilities used across multiple assertions
# ---------------------------------------------------------------------------
def extract_examples(text: str) -> list[str]:
    """Return all raw JSON strings embedded in ``[[Example]]`` blocks."""

    pattern = re.compile(r"\[\[Example(?: \d+)?]]\s+```json\s+(.*?)\s+```", re.DOTALL)
    return pattern.findall(text)


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

    # Header present exactly once at the very beginning.
    assert s.startswith("[[User Info]]"), "Missing or misplaced section header."

    # Top-level primitive fields.
    assert "name: Alice" in s
    assert "age: 30" in s

    # Nested model fields should appear on their own indented lines.
    assert "address:" in s and "street: Main" in s and "city: NY" in s

    # List items should be rendered using a dash ("-"). We do not enforce exact
    # wording – just verify that at least one bullet line is present for the
    # list field.
    assert "- :" in s, "List items should use dash bullet formatting."

    # Dict representation – key followed by value.
    assert "score:" in s


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
    assert "age: None" in s, "Fields with a value of None should be rendered as 'None'."


# ---------------------------------------------------------------------------
# ``generate_example_json``
# ---------------------------------------------------------------------------
def test_generate_example_json_complex():
    """The helper should create sensible placeholder values for a variety of
    field types including primitives, containers, and nested models.
    """

    example = LLMDataModel.generate_example_json(User)

    # Primitive placeholders.
    assert example["name"] == "example_string"
    assert example["age"] == 123  # Int example

    # Nested structure should itself be filled with placeholder primitives.
    address = example["address"]
    assert address == {"street": "example_string", "city": "example_string"}

    # List and dict placeholders.
    assert example["tags"] == ["example_string"]
    assert example["metadata"] == {"example_string": 123}


def test_generate_example_json_union_resolution():
    """For a simple ``int | str`` union the heuristic should choose the integer
    representation (less nested / primitive precedence).
    """

    DynModel = build_dynamic_llm_datamodel(int | str)
    example = LLMDataModel.generate_example_json(DynModel)

    assert example["value"] == 123, "Union heuristic should default to the integer example."


# ---------------------------------------------------------------------------
# ``simplify_json_schema``
# ---------------------------------------------------------------------------
def test_simplify_json_schema_contains_required_information():
    """A human-readable schema must at minimum contain each field name and a
    type description. A very strict equality assertion is intentionally
    avoided to keep the test resilient to non-breaking formatting tweaks.
    """

    schema_text = User.simplify_json_schema()

    # Expected top-level markers.
    assert schema_text.startswith("[[Schema]]")

    # Field names and basic type information.
    assert "name" in schema_text and "string" in schema_text
    assert "age" in schema_text and "integer" in schema_text
    assert "address" in schema_text and "nested object" in schema_text


# ---------------------------------------------------------------------------
# ``instruct_llm``
# ---------------------------------------------------------------------------
def test_instruct_llm_single_example_for_standard_model():
    """Models without unions should yield exactly one example block."""

    instr = User.instruct_llm()

    # Basic sanity – result block and schema must be present.
    assert "[[Result]]" in instr and "[[Schema]]" in instr

    examples = extract_examples(instr)
    assert len(examples) == 1

    # The JSON in the example block should be parsable.
    parsed = json.loads(examples[0])
    assert set(parsed.keys()) == set(User.model_fields.keys()) - {"section_header"}


def test_instruct_llm_multiple_examples_for_union():
    """A union containing two concrete alternatives should lead to two example
    blocks – one per alternative.
    """

    DynModel = build_dynamic_llm_datamodel(list[Address] | dict[str, int])
    instr = DynModel.instruct_llm()

    examples = extract_examples(instr)
    assert len(examples) == 2, "Expected one example per union alternative."

    # Ensure one example corresponds to the list[Address] variant and one to the dict variant.
    # We parse both JSON strings and inspect the *type* of the ``value`` field.
    v1_type = type(json.loads(examples[0])["value"])
    v2_type = type(json.loads(examples[1])["value"])

    assert {v1_type, v2_type} == {list, dict}, "Expected examples for both list and dict union variants."


def test_union_int_str():
    """Union of two primitive types should generate two example blocks."""

    model = build_dynamic_llm_datamodel(int | str)
    instr = model.instruct_llm()

    # Schema should contain both primitives.
    assert "integer or string" in instr or "string or integer" in instr

    # Two example blocks expected.
    ex_blocks = extract_examples(instr)
    assert len(ex_blocks) == 2, "Expected two examples for int | str union."

    # First example corresponds to `int` (value 123), second to `str`.
    assert "123" in ex_blocks[0] and "example_string" not in ex_blocks[0]
    assert "example_string" in ex_blocks[1]


def test_union_with_optional():
    """`str | None` should yield exactly one example (str)."""

    model = build_dynamic_llm_datamodel(str | None)
    instr = model.instruct_llm()
    ex_blocks = extract_examples(instr)

    assert len(ex_blocks) == 1, "Optional union should ignore None for examples."
    assert "example_string" in ex_blocks[0]


def test_user_defined_model_non_union():
    """Ensure instruct_llm works for normal user-defined models without 'value'."""

    instr = Thoughts.instruct_llm()
    # Should have a single example block illustrating list of strings.
    examples = extract_examples(instr)
    assert len(examples) == 1
    assert '"thoughts": [' in examples[0]


def test_nested_union_complex():
    """Union with complex nested types (list and dict containing BaseModel)."""

    union_t = list[Person] | dict[str, list[int]]
    model = build_dynamic_llm_datamodel(union_t)
    instr = model.instruct_llm()

    # Should have two example blocks
    ex_blocks = extract_examples(instr)
    assert len(ex_blocks) == 2

    # Example 1 should be list variant containing Person objects.
    list_ex = ex_blocks[0]
    assert list_ex.strip().startswith("{\n \"value\": ["), "First example is expected to be list variant."
    assert "\"name\":" in list_ex and "\"age\":" in list_ex

    # Example 2 should be dict[str, list[int]]
    dict_ex = ex_blocks[1]
    assert "\"value\": {" in dict_ex and "[\n   123\n  ]" in dict_ex


# ---------------------------------------------------------------------------
# Default ``validate`` and ``remedy``
# ---------------------------------------------------------------------------
def test_default_validate_and_remedy_are_noops():
    """The default implementation should return ``None`` to indicate the
    absence of custom validation or remedy logic.
    """

    m = Address(street="A", city="B")

    assert m.validate() is None
    assert m.remedy() is None
