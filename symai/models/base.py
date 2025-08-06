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
    )

    @model_validator(mode='before')
    @classmethod
    def validate_const_fields(cls, values):
        """Validate that const fields have their expected values."""
        for field_name, field_info in cls.model_fields.items():
            if cls._is_const_field(field_info):
                const_value = cls._get_const_value(field_info)
                if field_name in values and values[field_name] != const_value:
                    raise ValueError(f'{field_name} must be {const_value!r}')
        return values

    #@NOTE: JSON only supports string keys in objects. When a Python dict with integer keys
    # like {1: "a", 2: "b"} is serialized to JSON, it becomes {"1": "a", "2": "b"}.
    # These three methods handle the deserialization back to Python by converting string
    # keys to integers for fields typed as dict[int, ...]. This is a targeted solution
    # for this common JSON-to-Python type mismatch, not handling all possible key types.
    @model_validator(mode='before')
    @classmethod
    def convert_dict_int_keys(cls, values):
        """Convert string keys to integer keys for dict[int, ...] fields."""
        for field_name, field_info in cls.model_fields.items():
            if field_name in values and cls._has_dict_int_type(field_info.annotation):
                values[field_name] = cls._convert_keys_to_int(values[field_name])
        return values

    @classmethod
    def _has_dict_int_type(cls, field_type) -> bool:
        """Check if type contains dict[int, ...]"""
        origin = get_origin(field_type)
        if origin is dict:
            args = get_args(field_type)
            return args and args[0] is int
        elif origin in (Union, UnionType):
            return any(cls._has_dict_int_type(arg) for arg in get_args(field_type))
        return False

    @staticmethod
    def _convert_keys_to_int(value):
        """Convert dictionary string keys to integers if possible."""
        if not isinstance(value, dict):
            return value
        try:
            return {int(k): v for k, v in value.items()}
        except (ValueError, TypeError):
            return value

    @staticmethod
    def _is_union_type(field_type: Any) -> bool:
        """Check if a type is a Union."""
        return get_origin(field_type) in (Union, UnionType)

    @staticmethod
    def _is_optional_type(field_type: Any) -> bool:
        """Check if a type is Optional (Union with None)."""
        if not LLMDataModel._is_union_type(field_type):
            return False
        return type(None) in get_args(field_type)

    @staticmethod
    def _get_union_types(field_type: Any, exclude_none: bool = True) -> list:
        """Get types from a Union, optionally excluding None."""
        types = get_args(field_type)
        if exclude_none:
            return [t for t in types if t is not type(None)]
        return list(types)

    @staticmethod
    def _is_collection_type(field_type: Any) -> bool:
        """Check if a type is a collection (list, set, tuple, dict, etc.)."""
        origin = get_origin(field_type)
        return origin in (list, set, frozenset, tuple, dict) or field_type in (list, set, frozenset, tuple, dict)

    @staticmethod
    def _is_basemodel_type(field_type: Any) -> bool:
        """Check if a type is a BaseModel subclass."""
        origin = get_origin(field_type) or field_type
        return isinstance(origin, type) and issubclass(origin, BaseModel)

    @staticmethod
    def _is_const_field(field_info) -> bool:
        """Check if a field is a const field."""
        return (
            field_info.json_schema_extra and
            'const_value' in field_info.json_schema_extra
        )

    @staticmethod
    def _get_const_value(field_info):
        """Get the const value from a field."""
        return field_info.json_schema_extra.get('const_value')

    @staticmethod
    def _has_default_value(field_info) -> bool:
        """Check if a field has a default value."""
        return field_info.default != ... and field_info.default != PydanticUndefined

    @staticmethod
    def _get_default_value(field_info):
        """Get the default value from a field."""
        if LLMDataModel._has_default_value(field_info):
            return field_info.default
        elif field_info.default_factory is not None:
            return field_info.default_factory()
        return None

    def format_field(self, key: str, value: Any, indent: int = 0, visited: set = None) -> str:
        """Formats a field value for string representation, handling nested structures."""
        visited = visited or set()
        formatter = self._get_formatter_for_value(value)
        return formatter(key, value, indent, visited)

    def _get_formatter_for_value(self, value: Any):
        """Get the appropriate formatter function for a value type."""
        formatters = {
            type(None): self._format_none_field,
            Enum: self._format_enum_field,
            LLMDataModel: self._format_model_field,
            list: self._format_list_field,
            dict: self._format_dict_field,
        }

        for type_class, formatter in formatters.items():
            if type_class == type(None) and value is None:
                return formatter
            if type_class != type(None) and isinstance(value, type_class):
                return formatter

        return self._format_primitive_field

    def _format_none_field(self, key: str, value: Any, indent: int, visited: set) -> str:
        """Format a None value."""
        return f"{' ' * indent}{key}: None"

    def _format_enum_field(self, key: str, value: Enum, indent: int, visited: set) -> str:
        """Format an Enum value."""
        return f"{' ' * indent}{key}: {value.value}"

    def _format_model_field(self, key: str, value: "LLMDataModel", indent: int, visited: set) -> str:
        """Format a nested model field."""
        obj_id = id(value)
        indent_str = " " * indent
        if obj_id in visited:
            return f"{indent_str}{key}: <circular reference>"
        visited.add(obj_id)
        nested_str = value.__str__(indent, visited).strip()
        visited.discard(obj_id)
        return f"{indent_str}{key}:\n{indent_str}  {nested_str}"

    def _format_list_field(self, key: str, value: list, indent: int, visited: set) -> str:
        """Format a list field."""
        indent_str = " " * indent
        if not value:
            return f"{indent_str}{key}:\n"

        items = []
        for item in value:
            if isinstance(item, dict):
                dict_str = self.format_field("", item, indent + 2, visited).strip()
                items.append(f"{indent_str}  - :\n{indent_str}    {dict_str}")
            elif isinstance(item, list):
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

    def _format_dict_field(self, key: str, value: dict, indent: int, visited: set) -> str:
        """Format a dictionary field."""
        indent_str = " " * indent
        if not value:
            return f"{indent_str}{key}:\n"

        items = []
        for k, v in value.items():
            if isinstance(v, (dict, list, LLMDataModel)):
                nested_str = self.format_field(k, v, indent + 2, visited)
                items.append(nested_str)
            else:
                items.append(f"{indent_str}  {k}: {v}")
        return f"{indent_str}{key}:\n" + "\n".join(items) if key else "\n".join(items)

    def _format_primitive_field(self, key: str, value: Any, indent: int, visited: set) -> str:
        """Format a primitive field."""
        return f"{' ' * indent}{key}: {value}"

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
            )
        ]

        if field_list:
            fields = "\n".join(field_list) + "\n"
        else:
            fields = ""

        if self.section_header and indent == 0:
            header = f"{indent_str}[[{self.section_header}]]\n"
            return f"{header}{fields}"
        return fields

    def validate(self) -> bool:
        """Custom validation of the model."""
        return None

    def remedy(self):
        """Default remedy method for the model."""
        return None

    @classmethod
    @lru_cache(maxsize=128)
    def simplify_json_schema(cls) -> str:
        """Converts a schema from Pydantic's model_json_schema() into a simplified format."""
        schema = cls.model_json_schema()
        return cls._build_simplified_schema(schema)

    @classmethod
    def _build_simplified_schema(cls, schema: dict) -> str:
        """Build the simplified schema from raw schema."""
        properties = cls._extract_schema_properties(schema)
        definitions = cls._extract_schema_definitions(schema)

        extra_defs = set()
        main_schema = cls._format_schema_fields(properties, schema, definitions, 0, extra_defs)
        definitions_schema = cls._format_schema_definitions(definitions, extra_defs)

        return cls._compose_schema_output(main_schema, definitions_schema)

    @classmethod
    def _extract_schema_properties(cls, schema: dict) -> dict:
        """Extract properties from schema."""
        return schema.get("properties", {})

    @classmethod
    def _extract_schema_definitions(cls, schema: dict) -> dict:
        """Extract definitions from schema."""
        return schema.get("$defs", schema.get("definitions", {}))

    @classmethod
    def _format_schema_field(cls, name: str, field_schema: dict, required: bool,
                            definitions: dict, indent_level: int, visited: set = None) -> str:
        """Format a single schema field."""
        visited = visited or set()

        field_type = cls._resolve_field_type(field_schema, definitions)
        description = field_schema.get(
            "description", field_schema.get("title", "No description provided.")
        )
        is_required = "required" if required else "optional"
        indent = "  " * indent_level

        nested_description = ""
        if field_type.startswith("nested object"):
            nested_description = cls._format_nested_object_field(
                field_schema, definitions, indent_level, visited
            )
        elif field_type.startswith("array of nested object"):
            nested_description = cls._format_array_nested_object_field(
                field_schema, definitions, indent_level, visited
            )

        result = f'{indent}- "{name}" ({field_type}, {is_required}): {description}'
        if nested_description:
            result += f"\n{indent}  - Nested fields:\n{nested_description}"
        return result

    @classmethod
    def _format_nested_object_field(cls, field_schema: dict, definitions: dict,
                                   indent_level: int, visited: set) -> str:
        """Format nested object field description."""
        ref_name = field_schema.get("$ref", "").split("/")[-1]
        if ref_name in definitions and ref_name not in visited:
            visited.add(ref_name)
            return cls._format_schema_fields(
                definitions[ref_name].get("properties", {}),
                definitions[ref_name], definitions, indent_level + 1, set(), visited.copy()
            )
        return ""

    @classmethod
    def _format_array_nested_object_field(cls, field_schema: dict, definitions: dict,
                                         indent_level: int, visited: set) -> str:
        """Format array of nested object field description."""
        ref_name = field_schema.get("items", {}).get("$ref", "").split("/")[-1]
        if ref_name in definitions and ref_name not in visited:
            visited.add(ref_name)
            return cls._format_schema_fields(
                definitions[ref_name].get("properties", {}),
                definitions[ref_name], definitions, indent_level + 1, set(), visited.copy()
            )
        return ""

    @classmethod
    def _format_schema_fields(cls, properties: dict, schema: dict, definitions: dict,
                             indent_level: int, extra_defs: set, visited: set = None) -> str:
        """Format multiple schema fields."""
        visited = visited or set()
        required_fields = set(schema.get("required", []))
        lines = []

        for name, field_schema in properties.items():
            if name == "section_header":
                continue
            lines.append(
                cls._format_schema_field(
                    name, field_schema, name in required_fields,
                    definitions, indent_level, visited.copy()
                )
            )

            if extra_defs is not None:
                cls._collect_variant_definitions(field_schema, extra_defs)

        return "\n".join(lines)

    @classmethod
    def _collect_variant_definitions(cls, field_schema: dict, extra_defs: set) -> None:
        """Collect variant definitions from anyOf/oneOf fields."""
        variants = []
        if "anyOf" in field_schema:
            variants.extend(field_schema["anyOf"])
        if "oneOf" in field_schema:
            variants.extend(field_schema["oneOf"])

        for variant in variants:
            if "$ref" in variant:
                ref_name = variant["$ref"].split("/")[-1]
                extra_defs.add(ref_name)

    @classmethod
    def _resolve_field_type(cls, field_schema: dict, definitions: dict) -> str:
        """Resolve the type of a field from schema."""
        if "allOf" in field_schema:
            return cls._resolve_allof_type(field_schema, definitions)

        if "enum" in field_schema:
            return cls._format_enum_type(field_schema["enum"])

        if "anyOf" in field_schema:
            return cls._resolve_union_type(field_schema["anyOf"], definitions, " or ")
        if "oneOf" in field_schema:
            return cls._resolve_union_type(field_schema["oneOf"], definitions, " or ")

        if "type" in field_schema:
            return cls._resolve_basic_type(field_schema, definitions)

        if "$ref" in field_schema:
            return cls._resolve_reference_type(field_schema["$ref"])

        return "unknown"

    @classmethod
    def _resolve_allof_type(cls, field_schema: dict, definitions: dict) -> str:
        """Resolve allOf type schema."""
        if len(field_schema["allOf"]) != 1:
            return "unknown"

        inner = field_schema["allOf"][0]
        if "$ref" not in inner:
            return "unknown"

        ref_name = inner["$ref"].split("/")[-1]
        if ref_name not in definitions:
            return f"nested object ({ref_name})"

        ref_def = definitions[ref_name]
        if "enum" in ref_def:
            return cls._format_enum_type(ref_def["enum"])
        return f"nested object ({ref_name})"

    @classmethod
    def _format_enum_type(cls, enum_values: list) -> str:
        """Format enum type with values."""
        literal_values = ", ".join(map(repr, enum_values))
        return f"enum ({literal_values})"

    @classmethod
    def _resolve_union_type(cls, schemas: list, definitions: dict, separator: str) -> str:
        """Resolve union types (anyOf/oneOf)."""
        subtypes = [
            cls._resolve_field_type(subschema, definitions)
            for subschema in schemas
        ]
        return separator.join(subtypes)

    @classmethod
    def _resolve_basic_type(cls, field_schema: dict, definitions: dict) -> str:
        """Resolve basic type schema."""
        field_type = field_schema["type"]

        if field_type == "array":
            return cls._resolve_array_type(field_schema, definitions)

        if field_type == "object" and "additionalProperties" in field_schema:
            return cls._resolve_object_type(field_schema, definitions)

        return field_type

    @classmethod
    def _resolve_array_type(cls, field_schema: dict, definitions: dict) -> str:
        """Resolve array type schema."""
        items = field_schema.get("items", {})
        item_type = cls._resolve_field_type(items, definitions)
        return f"array of {item_type}"

    @classmethod
    def _resolve_object_type(cls, field_schema: dict, definitions: dict) -> str:
        """Resolve object type schema."""
        value_schema = field_schema.get("additionalProperties", {})
        if value_schema is True:
            return "object"
        value_type = cls._resolve_field_type(value_schema, definitions)
        return f"object of {value_type}"

    @classmethod
    def _resolve_reference_type(cls, ref: str) -> str:
        """Resolve reference type."""
        ref_name = ref.split("/")[-1]
        return f"nested object ({ref_name})"

    @classmethod
    def _format_schema_definitions(cls, definitions: dict, extra_defs: set) -> str:
        """Format schema definitions."""
        lines = []
        visited_defs = set()

        for name, definition in definitions.items():
            if name not in visited_defs:
                visited_defs.add(name)
                lines.append(f"- {name}:")
                if "enum" in definition:
                    enum_values = ", ".join(map(repr, definition["enum"]))
                    lines.append(f"  Enum values: {enum_values}")
                else:
                    formatted = cls._format_schema_fields(
                        definition.get("properties", {}),
                        definition, definitions, 1, extra_defs, {name}
                    )
                    lines.append(formatted)

        for desc in sorted(extra_defs):
            lines.append(f"- {desc}")

        return "\n".join(lines)

    @classmethod
    def _compose_schema_output(cls, main_schema: str, definitions_schema: str) -> str:
        """Compose the final schema output."""
        result = f"[[Schema]]\n{main_schema}"
        if definitions_schema:
            result += f"\n\n[[Definitions]]\n{definitions_schema}"
        return result

    @classmethod
    def generate_example_json(cls, model=None, visited_models=None) -> dict:
        """Generates an example JSON object from a Pydantic BaseModel."""
        if model is None:
            model = cls
        if visited_models is None:
            visited_models = set()

        return cls._generate_example_for_model(model, visited_models)

    @staticmethod
    def _generate_example_for_model(model: Type[BaseModel], visited_models: set) -> dict:
        """Generate example for a model, excluding section_header."""
        example = {}
        for field_name, model_field in model.model_fields.items():
            if field_name == "section_header":
                continue
            example[field_name] = LLMDataModel._generate_field_value(model_field, visited_models)
        return example

    @staticmethod
    def _generate_field_value(model_field, visited_models: set) -> Any:
        """Generate a value for a model field."""
        if LLMDataModel._has_default_value(model_field):
            return model_field.default
        elif model_field.default_factory is not None:
            # For example generation, we want to show structure even if default is empty
            # Check if default_factory would produce an empty container
            default_val = model_field.default_factory()
            if isinstance(default_val, (list, dict, set, tuple)) and len(default_val) == 0:
                # Generate example data instead of using empty default
                return LLMDataModel._generate_value_for_type(
                    model_field.annotation, visited_models
                )
            return default_val
        else:
            return LLMDataModel._generate_value_for_type(
                model_field.annotation, visited_models
            )

    @staticmethod
    def _generate_value_for_type(field_type: Any, visited_models: set) -> Any:
        """Generate a value for a specific type."""
        origin = get_origin(field_type) or field_type

        if isinstance(origin, type) and issubclass(origin, Enum):
            return list(origin)[0].value if list(origin) else "enum_value"

        if origin is Literal:
            return get_args(field_type)[0]

        if isinstance(origin, type) and issubclass(origin, BaseModel):
            model_name = field_type.__name__
            if model_name in visited_models:
                return {}
            visited_models.add(model_name)
            return LLMDataModel._generate_example_for_model(field_type, visited_models.copy())

        if LLMDataModel._is_union_type(field_type):
            return LLMDataModel._generate_union_value(field_type, visited_models)

        if LLMDataModel._is_collection_type(field_type):
            return LLMDataModel._generate_collection_value(field_type, visited_models)

        return LLMDataModel._generate_primitive_value(field_type)

    @staticmethod
    def _generate_union_value(field_type: Any, visited_models: set) -> Any:
        """Generate a value for a Union type."""
        subtypes = LLMDataModel._get_union_types(field_type, exclude_none=True)
        if not subtypes:
            return None
        return LLMDataModel._generate_value_for_type(subtypes[0], visited_models)

    @staticmethod
    def _generate_collection_value(field_type: Any, visited_models: set) -> Any:
        """Generate a value for a collection type."""
        origin = get_origin(field_type) or field_type

        if origin is list:
            return LLMDataModel._generate_list_value(field_type, visited_models)
        if origin is dict:
            return LLMDataModel._generate_dict_value(field_type, visited_models)
        if origin in (set, frozenset):
            return LLMDataModel._generate_set_value(field_type, visited_models)
        if origin is tuple:
            return LLMDataModel._generate_tuple_value(field_type, visited_models)

        return []

    @staticmethod
    def _generate_list_value(field_type: Any, visited_models: set) -> list:
        """Generate a value for a list type."""
        item_type = get_args(field_type)[0] if get_args(field_type) else Any

        if LLMDataModel._is_union_type(item_type):
            subtypes = LLMDataModel._get_union_types(item_type)
            return [
                LLMDataModel._generate_value_for_type(subtype, visited_models)
                for subtype in subtypes[:2]
            ]

        return [LLMDataModel._generate_value_for_type(item_type, visited_models)]

    @staticmethod
    def _generate_dict_value(field_type: Any, visited_models: set) -> dict:
        """Generate a value for a dict type."""
        key_type, value_type = get_args(field_type) if get_args(field_type) else (Any, Any)

        # Generate appropriate key based on the key type
        if key_type is int:
            example_key = 123
        elif key_type is float:
            example_key = 123.45
        elif key_type is bool:
            example_key = True
        elif key_type is tuple or get_origin(key_type) is tuple:
            # Handle tuple keys
            tuple_args = get_args(key_type) if get_args(key_type) else (str, int)
            example_key = tuple(LLMDataModel._generate_value_for_type(t, visited_models) for t in tuple_args)
        elif key_type is frozenset or get_origin(key_type) is frozenset:
            # Handle frozenset keys
            item_type = get_args(key_type)[0] if get_args(key_type) else str
            example_key = frozenset([LLMDataModel._generate_value_for_type(item_type, visited_models)])
        else:
            # Default to string for str, Any, or other types
            example_key = "example_string"

        return {example_key: LLMDataModel._generate_value_for_type(value_type, visited_models)}

    @staticmethod
    def _generate_set_value(field_type: Any, visited_models: set) -> list:
        """Generate a value for a set type (returns list for JSON serialization)."""
        item_type = get_args(field_type)[0] if get_args(field_type) else Any
        return [LLMDataModel._generate_value_for_type(item_type, visited_models)]

    @staticmethod
    def _generate_tuple_value(field_type: Any, visited_models: set) -> tuple:
        """Generate a value for a tuple type."""
        types = get_args(field_type)
        if types:
            return tuple(LLMDataModel._generate_value_for_type(t, visited_models) for t in types)
        return ("item1", "item2")

    @staticmethod
    def _generate_primitive_value(field_type: Any) -> Any:
        """Generate a value for a primitive type."""
        if field_type is str:
            return "example_string"
        if field_type is int:
            return 123
        if field_type is float:
            return 123.45
        if field_type is bool:
            return True
        if field_type is None or field_type is type(None):
            return None
        return "example_value"

    @classmethod
    @lru_cache(maxsize=128)
    def instruct_llm(cls):
        """Generate LLM instructions with schema and examples."""
        result_section = cls._generate_result_section()
        schema_section = cls._generate_schema_section()
        examples_section = cls._generate_examples_section()

        return f"""
{result_section}

{schema_section}

{examples_section}
"""

    @classmethod
    def _generate_result_section(cls) -> str:
        """Generate the result section of instructions."""
        return "[[Result]]\nReturn a JSON object with the following schema:"

    @classmethod
    def _generate_schema_section(cls) -> str:
        """Generate the schema section of instructions."""
        return cls.simplify_json_schema()

    @classmethod
    def _generate_examples_section(cls) -> str:
        """Generate the examples section of instructions."""
        examples = []
        user_fields = cls._find_non_header_fields()

        if cls._is_single_value_model(user_fields):
            examples = cls._generate_union_examples(user_fields["value"])
        elif len(user_fields) == 1:
            field_name, field = next(iter(user_fields.items()))
            examples = cls._generate_field_examples(field_name, field)
        else:
            examples = [cls._generate_single_example()]

        return cls._format_examples(examples)

    @classmethod
    def _find_non_header_fields(cls) -> dict:
        """Find all fields except section_header."""
        return {
            name: f for name, f in cls.model_fields.items()
            if name != "section_header"
        }

    @classmethod
    def _is_single_value_model(cls, fields: dict) -> bool:
        """Check if model has single 'value' field."""
        return "value" in fields and len(fields) == 1

    @classmethod
    def _generate_union_examples(cls, field) -> list:
        """Generate examples for union types."""
        annotation = field.annotation
        origin = get_origin(annotation)

        subtypes = (
            [a for a in get_args(annotation) if a is not type(None)]
            if origin in (Union, UnionType)
            else [annotation]
        )

        examples = []
        for subtype in subtypes:
            example = cls._create_example_with_type(subtype)
            examples.append(json.dumps(example, indent=1))
        return examples

    @classmethod
    def _generate_field_examples(cls, field_name: str, field) -> list:
        """Generate examples for a single field."""
        annotation = field.annotation
        origin = get_origin(annotation)

        subtypes = (
            [a for a in get_args(annotation) if a is not type(None)]
            if origin in (Union, UnionType)
            else [annotation]
        )

        examples = []
        for subtype in subtypes:
            temp_model = build_dynamic_llm_datamodel(subtype)
            value_example = cls.generate_example_json(temp_model, visited_models=set())["value"]
            examples.append(json.dumps({field_name: value_example}, indent=1))
        return examples

    @classmethod
    def _generate_single_example(cls):
        """Generate a single example for the model."""
        return json.dumps(cls.generate_example_json(cls), indent=1)

    @classmethod
    def _create_example_with_type(cls, subtype: Any) -> dict:
        """Create an example dict for a specific type."""
        submodel = build_dynamic_llm_datamodel(subtype)
        return submodel.generate_example_json()

    @classmethod
    def _format_examples(cls, examples: list) -> str:
        """Format examples into the final output string."""
        if len(examples) == 1:
            return f"[[Example]]\n```json\n{examples[0]}\n```"

        example_blocks = []
        for idx, ex in enumerate(examples, start=1):
            example_blocks.append(f"[[Example {idx}]]\n```json\n{ex}\n```")
        return "\n\n".join(example_blocks)


def build_dynamic_llm_datamodel(py_type: Any) -> Type[LLMDataModel]:
    """Dynamically create a subclass of LLMDataModel with a single 'value' field."""
    model_name = f"LLMDynamicDataModel_{hash(str(py_type)) & 0xFFFFFFFF:X}"

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
