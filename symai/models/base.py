import json
from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import Any, Literal, Type, Union, get_args, get_origin

from attr import dataclass
from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_core import PydanticUndefined


@dataclass
class LengthConstraint:
    field_name: str = None
    min_length: int = None
    max_length: int = None


@dataclass
class CustomConstraint:
    rule: str


def Const(value: str):
    return Field(default=value, json_schema_extra={'const_value': value})


class LLMDataModel(BaseModel):
    """
    A base class for Pydantic models that provides nicely formatted string output,
    suitable for LLM prompts, with support for nested models, lists, and optional section headers.
    """

    section_header: str = Field(
        default=None, exclude=True, frozen=True
    )  # Optional section header for top-level models

    @model_validator(mode='before')
    @classmethod
    def validate_const_fields(cls, values):
        """Validate that const fields have their expected values."""
        for field_name, field_info in cls.model_fields.items():
            if field_info.json_schema_extra and 'const_value' in field_info.json_schema_extra:
                const_value = field_info.json_schema_extra['const_value']
                if field_name in values and values[field_name] != const_value:
                    raise ValueError(f'{field_name} must be {const_value!r}')
        return values

    def format_field(self, key: str, value: Any, indent: int = 0, visited: set = None) -> str:
        """
        Formats a field value for string representation, handling nested structures.
        """
        if visited is None:
            visited = set()
        indent_str = " " * indent

        if value is None:
            return f"{indent_str}{key}: None"

        # Handle Enum values
        if isinstance(value, Enum):
            return f"{indent_str}{key}: {value.value}"

        if isinstance(value, LLMDataModel):
            obj_id = id(value)
            if obj_id in visited:
                return f"{indent_str}{key}: <circular reference>"
            visited.add(obj_id)
            nested_str = value.__str__(indent, visited).strip()
            visited.discard(obj_id)
            return f"{indent_str}{key}:\n{indent_str}  {nested_str}"

        if isinstance(value, list):
            if not value:  # Empty list
                return f"{indent_str}{key}:\n"
            items = []
            for item in value:
                if isinstance(item, dict):
                    # Recursively format nested dictionaries
                    dict_str = self.format_field("", item, indent + 2, visited).strip()
                    items.append(f"{indent_str}  - :\n{indent_str}    {dict_str}")
                elif isinstance(item, list):
                    # Recursively format nested lists
                    list_str = self.format_field("", item, indent + 2, visited).strip()
                    items.append(f"{indent_str}  - :\n{indent_str}    {list_str}")
                elif isinstance(item, LLMDataModel):
                    obj_id = id(item)
                    if obj_id in visited:
                        items.append(f"{indent_str}  - : <circular reference>")
                    else:
                        visited.add(obj_id)
                        item_str = item.__str__(indent + 2, visited).strip()
                        visited.discard(obj_id)
                        items.append(f"{indent_str}  - : {item_str}" if item_str else f"{indent_str}  - :")
                else:
                    items.append(f"{indent_str}  - : {item}" if item != "" else f"{indent_str}  - :")
            return f"{indent_str}{key}:\n" + "\n".join(items)

        if isinstance(value, dict):
            if not value:  # Empty dict
                return f"{indent_str}{key}:\n"
            items = []
            for k, v in value.items():
                # Recursively format nested values
                if isinstance(v, (dict, list, LLMDataModel)):
                    nested_str = self.format_field(k, v, indent + 2, visited)
                    items.append(nested_str)
                else:
                    items.append(f"{indent_str}  {k}: {v}")
            return f"{indent_str}{key}:\n" + "\n".join(items) if key else "\n".join(items)

        return f"{indent_str}{key}: {value}"

    def __str__(self, indent: int = 0, visited: set = None) -> str:
        """
        Converts the model into a formatted string for LLM prompts.
        Handles indentation for nested models and includes an optional section header.
        """
        if visited is None:
            visited = set()
        indent_str = " " * indent
        field_list = [
            self.format_field(name, getattr(self, name), indent + 2, visited)
            for name, field in type(self).model_fields.items()
            if (
                not getattr(field, "exclude", False)
                and not name == "section_header"
            )  # Exclude "exclude" fields
        ]

        if field_list:
            fields = "\n".join(field_list) + "\n"  # add line break at the end to separate from the next section
        else:
            fields = ""

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
    @lru_cache(maxsize=128)
    def simplify_json_schema(cls) -> str:
        """
        Converts a schema from Pydantic's model_json_schema() into a simplified, human-readable format
        with proper indentation for all levels of nested fields.

        Args:
            schema (dict): The JSON schema from Pydantic's model_json_schema().

        Returns:
            str: A simplified, human-readable schema description with proper indentation.
        """

        schema = cls.model_json_schema()

        def format_field(name, field_schema, required, definitions, indent_level, visited=None):
            """Formats a single field in the schema with proper indentation."""
            if visited is None:
                visited = set()

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
                if ref_name in definitions and ref_name not in visited:
                    visited.add(ref_name)
                    nested_description = simplify_fields(
                        definitions[ref_name], definitions, indent_level + 1, extra_defs, visited.copy()
                    )
            elif field_type.startswith("array of nested object"):
                ref_name = field_schema.get("items", {}).get("$ref", "").split("/")[-1]
                if ref_name in definitions and ref_name not in visited:
                    visited.add(ref_name)
                    nested_description = simplify_fields(
                        definitions[ref_name], definitions, indent_level + 1, extra_defs, visited.copy()
                    )

            # Build the field description
            result = f'{indent}- "{name}" ({field_type}, {is_required}): {description}'
            if nested_description:
                result += f"\n{indent}  - Nested fields:\n{nested_description}"
            return result

        def resolve_field_type(field_schema, definitions):
            """Resolves the type of a field, including arrays and $ref references."""
            # Handle allOf with $ref (common for enums)
            if "allOf" in field_schema and len(field_schema["allOf"]) == 1:
                inner = field_schema["allOf"][0]
                if "$ref" in inner:
                    ref_name = inner["$ref"].split("/")[-1]
                    if ref_name in definitions:
                        ref_def = definitions[ref_name]
                        if "enum" in ref_def:
                            literal_values = ", ".join(map(repr, ref_def["enum"]))
                            return f"enum ({literal_values})"
                        return f"nested object ({ref_name})"
            if "enum" in field_schema:
                literal_values = ", ".join(map(repr, field_schema["enum"]))
                return f"enum ({literal_values})"
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
                # Handle dictionaries / mappings via `additionalProperties`
                # Pydantic represents ``dict[key_type, value_type]`` as a JSON schema object
                # with an ``additionalProperties`` entry that contains the value schema.
                # If we detect such a structure we propagate the value type information so
                # that the human-readable schema makes clear what the object contains.
                if field_type == "object" and "additionalProperties" in field_schema:
                    value_schema = field_schema.get("additionalProperties", {})
                    # If ``additionalProperties`` is simply ``true`` we treat it as a plain object.
                    if value_schema is True:
                        return "object"

                    value_type = resolve_field_type(value_schema, definitions)
                    return f"object of {value_type}"
                return field_type
            if "$ref" in field_schema:
                ref_name = field_schema["$ref"].split("/")[-1]
                return f"nested object ({ref_name})"
            return "unknown"

        def simplify_fields(schema, definitions, indent_level=0, extra_defs: set[str] | None = None, visited=None):
            """Simplifies the properties of a schema with proper indentation."""
            if visited is None:
                visited = set()
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
                        visited.copy(),
                    )
                )

                # Collect union/oneOf/anyOf alternative type descriptions that are not $ref'd objects
                if extra_defs is not None:
                    variants = []
                    if "anyOf" in field_schema:
                        variants.extend(field_schema["anyOf"])
                    if "oneOf" in field_schema:
                        variants.extend(field_schema["oneOf"])

                    for variant in variants:
                        # Skip definitions that are already objects referenced via $ref
                        if "$ref" in variant:
                            continue
                        desc = resolve_field_type(variant, definitions)
                        if desc:
                            extra_defs.add(desc)
            return "\n".join(lines)

        def simplify_definitions(definitions, extra_defs: set[str]):
            """Simplifies the definitions for all referenced models."""
            lines = []
            visited_defs = set()
            for name, definition in definitions.items():
                if name not in visited_defs:
                    visited_defs.add(name)
                    lines.append(f"- {name}:")
                    # Check if this is an enum definition
                    if "enum" in definition:
                        # Format enum values
                        enum_values = ", ".join(map(repr, definition["enum"]))
                        lines.append(f"  Enum values: {enum_values}")
                    else:
                        # Regular object with properties
                        lines.append(simplify_fields(definition, definitions, indent_level=1, extra_defs=extra_defs, visited={name}))

            # Append pseudo-definitions for union alternatives that are primitives or simple arrays/objects
            for desc in sorted(extra_defs):
                lines.append(f"- {desc}")
            return "\n".join(lines)


        # Extract and simplify the main schema
        definitions = schema.get("$defs", schema.get("definitions", {}))
        # Collect additional union alternatives that are not objects with $ref
        extra_defs: set[str] = set()

        main_schema = simplify_fields(schema, definitions, indent_level=0, extra_defs=extra_defs)
        result = f"[[Schema]]\n{main_schema}"

        # Extract and simplify definitions
        definitions_schema = simplify_definitions(definitions, extra_defs)
        if definitions_schema:
            result += f"\n\n[[Definitions]]\n{definitions_schema}"

        # Combine the main schema and definitions
        return result

    @staticmethod
    def generate_example_json(model: Type[BaseModel], visited_models=None) -> dict:
        """
        Generates an example JSON object from a Pydantic BaseModel, recursively handling nested models,
        including cases where a field is a list with mixed types (e.g., Union[TypeA, TypeB]).

        Args:
            model (Type[BaseModel]): The Pydantic BaseModel class.

        Returns:
            dict: An example JSON object.
        """

        def resolve_field_value(field_annotation: Any, visited_models=None) -> Any:
            """Recursively resolves the value of a field."""
            if visited_models is None:
                visited_models = set()

            origin = get_origin(field_annotation)
            if origin is None:
                origin = field_annotation

            # Handle Enum
            if isinstance(origin, type) and issubclass(origin, Enum):
                # Return the first enum value's actual value
                return list(origin)[0].value if list(origin) else "enum_value"

            # Handle Literal
            if origin is Literal:
                return get_args(field_annotation)[0]  # Use the first allowed value as the example

            # Handle nested BaseModel
            if isinstance(origin, type) and issubclass(origin, BaseModel):
                # Avoid infinite recursion for self-referential models
                model_name = field_annotation.__name__
                if model_name in visited_models:
                    return [] if "list" in str(field_annotation).lower() else {}
                visited_models.add(model_name)
                result = LLMDataModel.generate_example_json(field_annotation, visited_models.copy())
                return result

            # Handle List
            if origin is list:
                item_type = (
                    get_args(field_annotation)[0] if get_args(field_annotation) else Any
                )
                # If the item type is a Union, generate examples for each possible type
                if get_origin(item_type) in [Union, UnionType]:
                    subtypes = get_args(item_type)
                    return [
                        resolve_field_value(subtype, visited_models)
                        for subtype in subtypes
                        if subtype is not type(None)
                    ]
                return [resolve_field_value(item_type, visited_models)]  # Single type in the list

            # Handle Dict
            if origin is dict:
                key_type, value_type = (
                    get_args(field_annotation)
                    if get_args(field_annotation)
                    else (str, Any)
                )
                return {resolve_field_value(key_type, visited_models): resolve_field_value(value_type, visited_models)}

            # Handle Set
            if origin is set:
                item_type = (
                    get_args(field_annotation)[0] if get_args(field_annotation) else Any
                )
                # Return as list for JSON serialization (sets aren't JSON serializable)
                return [resolve_field_value(item_type, visited_models), resolve_field_value(item_type, visited_models)]

            # Handle Frozenset
            if origin is frozenset:
                item_type = (
                    get_args(field_annotation)[0] if get_args(field_annotation) else Any
                )
                # Return as list for JSON serialization (frozensets aren't JSON serializable)
                return [resolve_field_value(item_type, visited_models), resolve_field_value(item_type, visited_models)]

            # Handle Tuple
            if origin is tuple:
                type_args = get_args(field_annotation)
                if type_args:
                    # Return as list for JSON serialization (tuples aren't JSON serializable)
                    return [resolve_field_value(t, visited_models) for t in type_args]
                return ["example_value"]

            # Handle Union (e.g., Union[TypeA, TypeB, None])
            if origin in [Union, UnionType]:
                # Filter out `None` so we only consider real alternatives.
                subtypes = [s for s in get_args(field_annotation) if s is not type(None)]

                # Simply use the first non-None type in the Union
                chosen_subtype = subtypes[0] if subtypes else None
                return resolve_field_value(chosen_subtype, visited_models) if chosen_subtype else None

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
            # Use default value if available (including None)
            if model_field.default != ...:
                if model_field.default != PydanticUndefined:
                    example[field_name] = model_field.default
                else:
                    example[field_name] = resolve_field_value(field_annotation, visited_models)
            elif model_field.default_factory is not None:
                example[field_name] = model_field.default_factory()
            else:
                example[field_name] = resolve_field_value(field_annotation, visited_models)

        return example

    ########################################
    # Helper methods for generating LLM instructions
    ########################################

    @classmethod
    @lru_cache(maxsize=128)
    def instruct_llm(cls):
        # ------------------------------------------------------------------
        # 1. Human-readable schema
        # ------------------------------------------------------------------
        human_readable_schema = cls.simplify_json_schema()

        # ------------------------------------------------------------------
        # 2. Generate example(s). If the top-level `value` annotation is a
        #    Union we create one example per alternative.
        # ------------------------------------------------------------------
        examples: list[str] = []

        # Determine how to construct examples based on the model structure
        # ------------------------------------------------------------------
        user_fields = {
            name: f for name, f in cls.model_fields.items() if name != "section_header"
        }

        def _append_example(example_dict: dict):
            examples.append(json.dumps(example_dict, indent=1))

        if (
            # Classic dynamic model created via `build_dynamic_llm_datamodel`
            "value" in user_fields
            and len(user_fields) == 1
        ):
            value_annotation = user_fields["value"].annotation
            origin = get_origin(value_annotation)

            subtypes = (
                [a for a in get_args(value_annotation) if a is not type(None)]
                if origin in (Union, UnionType)
                else [value_annotation]
            )

            for subtype in subtypes:
                submodel = build_dynamic_llm_datamodel(subtype)
                example_dict = cls.generate_example_json(submodel)
                _append_example(example_dict)
        elif len(user_fields) == 1:
            # Single-field user-defined model; might still be Union.
            field_name, field = next(iter(user_fields.items()))
            annotation = field.annotation
            origin = get_origin(annotation)

            subtypes = (
                [a for a in get_args(annotation) if a is not type(None)]
                if origin in (Union, UnionType)
                else [annotation]
            )

            for subtype in subtypes:
                # Generate example value for this subtype and embed in dict
                temp_model = build_dynamic_llm_datamodel(subtype)
                value_example = cls.generate_example_json(temp_model)["value"]
                _append_example({field_name: value_example})
        else:
            # Complex model with multiple fields – fall back to single example
            _append_example(cls.generate_example_json(cls))

        # ------------------------------------------------------------------
        # 3. Assemble instruction
        # ------------------------------------------------------------------
        example_blocks = []
        if len(examples) == 1:
            example_blocks.append("[[Example]]\n```json\n" + examples[0] + "\n´´´")
        else:
            for idx, ex in enumerate(examples, start=1):
                example_blocks.append(
                    f"[[Example {idx}]]\n```json\n{ex}\n´´´"
                )

        examples_section = "\n\n".join(example_blocks)

        return f"""
[[Result]]
Return a JSON object with the following schema:

{human_readable_schema}

{examples_section}
"""

def build_dynamic_llm_datamodel(py_type: Any) -> Type[LLMDataModel]:
    """Dynamically create a subclass of ``LLMDataModel`` that contains a single
    field called ``value`` annotated with *py_type*.

    Parameters
    ----------
    py_type : Any
        A Python or ``typing`` type that is supported by Pydantic (e.g.
        ``str``, ``list[str]``, ``dict[str, int]``).

    Returns
    -------
    Type[LLMDataModel]
        A newly created subclass of ``LLMDataModel`` with one
        required field named ``value``.
    """

    model_name = f"LLMDynamicDataModel_{hash(str(py_type)) & 0xFFFFFFFF:X}"  # deterministic but short name

    # `create_model` returns a new Pydantic model class
    model: Type[LLMDataModel] = create_model(
        model_name,
        __base__=LLMDataModel,
        value=(
            py_type,
            Field(
                ...,
                description="This is a dynamically generated data model. This description is general. " +
                "If you're dealing with a complex type, or nested types in combination with unions, make sure you " +
                "understand the instructions provided in the prompt, and select the appropriate data model based on the " +
                "type at hand, as described in the schema section."
            )
        ),
    )

    return model
