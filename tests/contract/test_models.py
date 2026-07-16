import pytest
from pydantic import Field, ValidationError

from symai.contract.models import Const, LLMDataModel, build_dynamic_llm_datamodel


class Address(LLMDataModel):
    street: str = Field(description="Street name")
    city: str = Field(description="City name")


class Review(LLMDataModel):
    text: str = Field(description="Review text")
    address: Address
    tags: list[str]
    score: int | None = None


def test_data_model_renders_nested_prompt_input() -> None:
    review = Review(
        text="Useful",
        address=Address(street="Main", city="Oxford"),
        tags=["clear", "short"],
        section_header="Review",
    )

    rendered = review.render()

    assert rendered.startswith("[[Review]]")
    assert "text: Useful" in rendered
    assert "street: Main" in rendered
    assert "- clear" in rendered
    assert "score: None" in rendered
    assert str(review) == rendered


def test_data_model_instructions_include_descriptions_schema_and_valid_example() -> None:
    instructions = Review.instruct_llm()

    assert "[[Schema]]" in instructions
    assert "Review text" in instructions
    assert "Street name" in instructions
    assert "[[Example]]" in instructions
    example = instructions.rsplit("```json\n", maxsplit=1)[1].split("\n```", maxsplit=1)[0]
    assert Review.model_validate_json(example)


def test_const_enforces_its_declared_value() -> None:
    class Tagged(LLMDataModel):
        kind: str = Const("review")

    assert Tagged().kind == "review"
    with pytest.raises(ValidationError):
        Tagged(kind="other")


def test_dynamic_data_model_wraps_supported_python_types() -> None:
    model_type = build_dynamic_llm_datamodel(list[int])
    value = model_type(value=[1, 2])

    assert value.value == [1, 2]
    assert model_type.model_validate_json('{"value":[3]}').value == [3]
    assert model_type.model_json_schema()["type"] == "object"
