import json
from functools import lru_cache
from types import UnionType
from typing import Any, Literal, Type, Union, get_args, get_origin

from attr import dataclass
from pydantic import BaseModel, Field, create_model


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
            for name, field in type(self).model_fields.items()
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
                        definitions[ref_name], definitions, indent_level + 1, extra_defs
                    )
            elif field_type.startswith("array of nested object"):
                ref_name = field_schema.get("items", {}).get("$ref", "").split("/")[-1]
                if ref_name in definitions:
                    nested_description = simplify_fields(
                        definitions[ref_name], definitions, indent_level + 1, extra_defs
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

        def simplify_fields(schema, definitions, indent_level=0, extra_defs: set[str] | None = None):
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
            for name, definition in definitions.items():
                lines.append(f"- {name}:")
                lines.append(simplify_fields(definition, definitions, indent_level=1, extra_defs=extra_defs))

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
                # Filter out `None` so we only consider real alternatives.
                subtypes = [s for s in get_args(field_annotation) if s is not type(None)]

                # Heuristic: prefer less-nested and more primitive representations, in the
                # following precedence order. This reduces bias towards complex objects and
                # makes it more likely that every union alternative is, at some point, used
                # as an example across different combinations.
                # We prefer examples that best showcase structure:
                # 0 – alternatives containing BaseModel (either directly or as value in dict/list)
                # 1 – dict variants
                # 2 – list variants
                # 3 – everything else (primitives)
                def contains_basemodel(annotation: Any) -> bool:
                    """Recursively check whether *annotation* contains a pydantic BaseModel."""
                    orig = get_origin(annotation)
                    if orig is None:
                        orig = annotation
                    if isinstance(orig, type) and issubclass(orig, BaseModel):
                        return True
                    if orig in (list, tuple, dict, Union, UnionType):
                        for sub in get_args(annotation):
                            if contains_basemodel(sub):
                                return True
                    return False

                def priority(t: Any) -> int:
                    if contains_basemodel(t):
                        return 0
                    t_origin = get_origin(t) or t
                    if t_origin is dict:
                        return 1
                    if t_origin is list:
                        return 2
                    return 3

                chosen_subtype = sorted(subtypes, key=priority)[0] if subtypes else None
                return resolve_field_value(chosen_subtype) if chosen_subtype else None

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
        # ------------------------------------------------------------------
        # 1. Human-readable schema
        # ------------------------------------------------------------------
        human_readable_schema = model.simplify_json_schema()

        # ------------------------------------------------------------------
        # 2. Generate example(s). If the top-level `value` annotation is a
        #    Union we create one example per alternative (excluding None).
        # ------------------------------------------------------------------
        examples: list[str] = []

        # Determine how to construct examples based on the model structure
        # ------------------------------------------------------------------
        user_fields = {
            name: f for name, f in model.model_fields.items() if name != "section_header"
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
                example_dict = LLMDataModel.generate_example_json(submodel)
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
                value_example = LLMDataModel.generate_example_json(temp_model)["value"]
                _append_example({field_name: value_example})
        else:
            # Complex model with multiple fields – fall back to single example
            _append_example(LLMDataModel.generate_example_json(model))

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
