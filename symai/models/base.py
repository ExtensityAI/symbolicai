import json
from types import UnionType
from typing import Any, Literal, Type, Union, get_args, get_origin

from attr import dataclass
from pydantic import BaseModel, Field


@dataclass
class LengthConstraint:
    field_name: str
    min_length: int
    max_length: int

@dataclass
class CustomConstraint:
    rule: str


def Const(value: str):
    return Field(default=value, frozen=True, init_var=False)


class LLMDataModel(BaseModel):
    """
    A base class for Pydantic models that provides nicely formatted string output,
    suitable for LLM prompts, with support for nested models, lists, and optional section headers.
    """

    section_header: str = Field(
        default=None, exclude=True, frozen=True
    )  # Optional section header for top-level models

    def format_field(self, key: str, value: Any, indent: int = 0) -> str:
        """
        Formats a single field for output. Handles nested models, lists, and dictionaries.
        """
        indent_str = " " * indent

        if isinstance(value, LLMDataModel):
            nested_str = value.__str__(indent).strip()
            return f"{indent_str}{key}:\n{indent_str}  {nested_str}"

        if isinstance(value, list):
            items = [f"{indent_str}  - {self.format_field('', item, indent).strip()}"
                    for item in value]
            return f"{indent_str}{key}:\n" + "\n".join(items)

        if isinstance(value, dict):
            items = [f"{indent_str}  {k}: {self.format_field('', v, indent).strip()}"
                    for k, v in value.items()]
            return f"{indent_str}{key}:\n" + "\n".join(items)

        return f"{indent_str}{key}: {value}"

    def __str__(self, indent: int = 0) -> str:
        """
        Converts the model into a formatted string for LLM prompts.
        Handles indentation for nested models and includes an optional section header.
        """
        indent_str = " " * indent
        fields = "\n".join(
            self.format_field(name, getattr(self, name), indent + 2)
            for name, field in self.model_fields.items()
            if (
                getattr(self, name, None) is not None
                and not getattr(field, "exclude", False)
                and not name == "section_header"
            )  # Exclude None values and "exclude" fields
        )
        fields += "\n"  # add line break at the end to separate from the next section

        if self.section_header and indent == 0:
            header = f"{indent_str}[[{self.section_header}]]\n"
            return f"{header}{fields}"
        return fields

    ########################################
    # Local validation method
    ########################################

    def validate(self) -> bool:
        """
        Custom validation of the model. Raise a TypeValidationError with remedy instructions if the model is invalid.
        """
        return None

    ########################################
    # Default remedy method
    ########################################
    def remedy(self):
        """Default remedy method for the model."""
        return None

    ########################################
    # Helper methods for generating LLM instructions
    ########################################
    @classmethod
    def simplify_json_schema(model) -> str:
        """
        Converts a schema from Pydantic's model_json_schema() into a simplified, human-readable format
        with proper indentation for all levels of nested fields.

        Args:
            schema (dict): The JSON schema from Pydantic's model_json_schema().

        Returns:
            str: A simplified, human-readable schema description with proper indentation.
        """

        schema = model.model_json_schema()

        def format_field(name, field_schema, required, definitions, indent_level):
            """Formats a single field in the schema with proper indentation."""
            field_type = resolve_field_type(field_schema, definitions)
            description = field_schema.get(
                "description", field_schema.get("title", "No description provided.")
            )
            is_required = "required" if required else "optional"
            indent = "  " * indent_level

            # Add nested fields for objects or arrays of objects
            nested_description = ""
            if field_type.startswith("nested object"):
                ref_name = field_schema.get("$ref", "").split("/")[-1]
                if ref_name in definitions:
                    nested_description = simplify_fields(
                        definitions[ref_name], definitions, indent_level + 1
                    )
            elif field_type.startswith("array of nested object"):
                ref_name = field_schema.get("items", {}).get("$ref", "").split("/")[-1]
                if ref_name in definitions:
                    nested_description = simplify_fields(
                        definitions[ref_name], definitions, indent_level + 1
                    )

            # Build the field description
            result = f'{indent}- "{name}" ({field_type}, {is_required}): {description}'
            if nested_description:
                result += f"\n{indent}  - Nested fields:\n{nested_description}"
            return result

        def resolve_field_type(field_schema, definitions):
            """Resolves the type of a field, including arrays and $ref references."""
            if "enum" in field_schema:
                literal_values = ", ".join(map(repr, field_schema["enum"]))
                return f"one of ({literal_values})"
            if "anyOf" in field_schema:
                subtypes = [
                    resolve_field_type(subschema, definitions)
                    for subschema in field_schema["anyOf"]
                ]
                return " or ".join(subtypes)
            if "oneOf" in field_schema:
                subtypes = [
                    resolve_field_type(subschema, definitions)
                    for subschema in field_schema["oneOf"]
                ]
                return " or ".join(subtypes)
            if "type" in field_schema:
                field_type = field_schema["type"]
                if field_type == "array":
                    items = field_schema.get("items", {})
                    item_type = resolve_field_type(items, definitions)
                    return f"array of {item_type}"
                return field_type
            if "$ref" in field_schema:
                ref_name = field_schema["$ref"].split("/")[-1]
                return f"nested object ({ref_name})"
            return "unknown"

        def simplify_fields(schema, definitions, indent_level=0):
            """Simplifies the properties of a schema with proper indentation."""
            properties = schema.get("properties", {})
            required_fields = set(schema.get("required", []))
            lines = []
            for name, field_schema in properties.items():
                if name == "section_header":
                    continue
                lines.append(
                    format_field(
                        name,
                        field_schema,
                        name in required_fields,
                        definitions,
                        indent_level,
                    )
                )
            return "\n".join(lines)

        def simplify_definitions(definitions):
            """Simplifies the definitions for all referenced models."""
            lines = []
            for name, definition in definitions.items():
                lines.append(f"- {name}:")
                lines.append(simplify_fields(definition, definitions, indent_level=1))
            return "\n".join(lines)


        # Extract and simplify the main schema
        definitions = schema.get("$defs", schema.get("definitions", {}))
        main_schema = simplify_fields(schema, definitions, indent_level=0)
        result = f"[[Schema]]\n{main_schema}"

        # Extract and simplify definitions
        definitions_schema = simplify_definitions(definitions)
        if definitions_schema:
            result += f"\n\n[[Definitions]]\n{definitions_schema}"

        # Combine the main schema and definitions
        return result

    @staticmethod
    def generate_example_json(model: Type[BaseModel]) -> dict:
        """
        Generates an example JSON object from a Pydantic BaseModel, recursively handling nested models,
        including cases where a field is a list with mixed types (e.g., Union[TypeA, TypeB]).

        Args:
            model (Type[BaseModel]): The Pydantic BaseModel class.

        Returns:
            dict: An example JSON object.
        """

        def resolve_field_value(field_annotation: Any) -> Any:
            """Recursively resolves the value of a field."""
            origin = get_origin(field_annotation)
            if origin is None:
                origin = field_annotation

            # Handle Literal
            if origin is Literal:
                return get_args(field_annotation)[
                    0
                ]  # Use the first allowed value as the example

            # Handle nested BaseModel
            if isinstance(origin, type) and issubclass(origin, BaseModel):
                return LLMDataModel.generate_example_json(field_annotation)

            # Handle List
            if origin is list:
                item_type = (
                    get_args(field_annotation)[0] if get_args(field_annotation) else Any
                )
                # If the item type is a Union, generate examples for each possible type
                if get_origin(item_type) in [Union, UnionType]:
                    subtypes = get_args(item_type)
                    return [
                        resolve_field_value(subtype)
                        for subtype in subtypes
                        if subtype is not type(None)
                    ]
                return [resolve_field_value(item_type)]  # Single type in the list

            # Handle Dict
            if origin is dict:
                key_type, value_type = (
                    get_args(field_annotation)
                    if get_args(field_annotation)
                    else (str, Any)
                )
                return {resolve_field_value(key_type): resolve_field_value(value_type)}

            # Handle Union (e.g., Union[TypeA, TypeB, None])
            if origin in [Union, UnionType]:
                subtypes = get_args(field_annotation)
                # Include an example for each subtype except NoneType
                for subtype in subtypes:
                    if subtype is not type(None):  # Skip NoneType
                        return resolve_field_value(subtype)
                return None

            # Primitive types
            if field_annotation is str:
                return "example_string"
            if field_annotation is int:
                return 123
            if field_annotation is float:
                return 123.45
            if field_annotation is bool:
                return True
            if field_annotation is None or field_annotation is type(None):
                return None

            # Fallback for unknown types
            return "example_value"

        # Generate the example JSON object
        example = {}
        for field_name, model_field in model.model_fields.items():
            if field_name == "section_header":
                continue
            field_annotation = model_field.annotation
            example[field_name] = resolve_field_value(field_annotation)

        return example

    ########################################
    # Helper methods for generating LLM instructions
    ########################################

    @classmethod
    def instruct_llm(model):
        # simplify the schema and generate LLM instructions
        human_readable_schema = model.simplify_json_schema()

        # generate an example
        example_json = json.dumps(LLMDataModel.generate_example_json(model), indent=1)

        return f"""
[[Result]]
Return a JSON object with the following schema:

{human_readable_schema}

[[Example]]
```json
{example_json}
´´´
"""
