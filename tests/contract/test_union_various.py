import re

from symai.models.base import LLMDataModel, build_dynamic_llm_datamodel


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def extract_examples(text: str) -> list[str]:
    """Return a list of the raw JSON strings contained in example blocks."""

    pattern = re.compile(r"\[\[Example(?: \d+)?]]\s+```json\s+(.*?)\s+´´´", re.DOTALL)
    return pattern.findall(text)


class Person(LLMDataModel):
    name: str
    age: int | None


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
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

    class Thoughts(LLMDataModel):
        thoughts: list[str]

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
