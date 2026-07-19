"""Parity tests for the vendored transform_schema (symai/strategy.py).

The function was vendored from anthropic-sdk-python 0.111.0 (Apache-2.0) so the SDK
could leave the dependency tree. These tests pin the exact SDK behavior on schemas
that exercise every branch: $defs/$ref nesting, anyOf, ge/le folding into description,
unsupported formats, additionalProperties=False, minItems rules.
"""

from pydantic import Field, RootModel

from symai.models import LLMDataModel
from symai.strategy import transform_schema


class Address(LLMDataModel):
    street: str = Field(description="Street name")
    zip_code: int = Field(description="Postal code", ge=1000, le=99999)


class Person(LLMDataModel):
    name: str
    age: int | None = None
    addresses: list[Address] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list, min_length=2)


def test_transform_schema_parity_with_sdk():
    schema = transform_schema(Person)

    # object envelope
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    # $defs kept and recursively transformed
    address = schema["$defs"]["Address"]
    assert address["additionalProperties"] is False
    assert address["required"] == ["street", "zip_code"]

    # ge/le are not in the Anthropic subset: folded into the description
    zip_code = address["properties"]["zip_code"]
    assert zip_code["type"] == "integer"
    assert "minimum" not in zip_code
    assert "maximum" not in zip_code
    assert zip_code["description"] == "Postal code\n\n{maximum: 99999, minimum: 1000}"

    # anyOf for optional fields
    age = schema["properties"]["age"]
    assert "anyOf" in age

    # nested $ref array items
    assert schema["properties"]["addresses"]["items"] == {"$ref": "#/$defs/Address"}

    # minItems > 1 is unsupported: folded into the description
    tags = schema["properties"]["tags"]
    assert "minItems" not in tags
    assert "{minItems: 2}" in tags.get("description", "")


def test_transform_schema_root_model_ref_keeps_defs():
    class AddressList(RootModel):
        root: list[Address]

    schema = transform_schema(AddressList.model_json_schema())
    # pydantic v2 RootModel emits type:array + items.$ref; $defs must survive the transform
    assert schema["items"] == {"$ref": "#/$defs/Address"}
    assert "Address" in schema["$defs"]


def test_transform_schema_string_format_rules():
    class Event(LLMDataModel):
        name: str
        timestamp: str = Field(format="date-time")
        weird: str = Field(format="not-a-real-format")

    schema = transform_schema(Event)
    props = schema["properties"]
    assert props["timestamp"]["format"] == "date-time"
    # unsupported format is folded into the description instead
    assert "format" not in props["weird"] or props["weird"].get("format") == "text"
