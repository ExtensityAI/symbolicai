import json
from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from attr import dataclass
from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_core import PydanticUndefined

from ..utils import UserMessage


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

    _MAX_RECURSION_DEPTH = 50

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
                    UserMessage(f'{field_name} must be {const_value!r}', raise_with=ValueError)
        return values

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

    def format_field(self, key: str, value: Any, indent: int = 0, visited: set | None = None, depth: int = 0) -> str:
        """Formats a field value for string representation, handling nested structures."""
        visited = visited or set()
        formatter = self._get_formatter_for_value(value)
        return formatter(key, value, indent, visited, depth)

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
            if type_class is type(None) and value is None:
                return formatter
            if type_class is not type(None) and isinstance(value, type_class):
                return formatter

        return self._format_primitive_field

    def _format_none_field(self, key: str, _value: Any, indent: int, _visited: set, _depth: int) -> str:
        """Format a None value."""
        return f"{' ' * indent}{key}: None"

    def _format_enum_field(self, key: str, value: Enum, indent: int, _visited: set, _depth: int) -> str:
        """Format an Enum value."""
        return f"{' ' * indent}{key}: {value.value}"

    def _format_model_field(self, key: str, value: "LLMDataModel", indent: int, visited: set, depth: int) -> str:
        """Format a nested model field."""
        obj_id = id(value)
        indent_str = " " * indent
        if obj_id in visited:
            return f"{indent_str}{key}: <circular reference>"
        if depth >= self._MAX_RECURSION_DEPTH:
            return f"{indent_str}{key}: <max depth reached>"
        visited.add(obj_id)
        nested_str = value.__str__(indent, visited, depth + 1).strip()
        visited.discard(obj_id)
        return f"{indent_str}{key}:\n{indent_str}  {nested_str}"

    def _format_list_field(self, key: str, value: list, indent: int, visited: set, depth: int) -> str:
        """Format a list field."""
        indent_str = " " * indent
        if not value:
            return f"{indent_str}{key}:\n"

        items = []
        for item in value:
            if isinstance(item, dict):
                dict_str = self.format_field("", item, indent + 2, visited, depth + 1).strip()
                items.append(f"{indent_str}  - :\n{indent_str}    {dict_str}")
            elif isinstance(item, list):
                list_str = self.format_field("", item, indent + 2, visited, depth + 1).strip()
                items.append(f"{indent_str}  - :\n{indent_str}    {list_str}")
            elif isinstance(item, LLMDataModel):
                obj_id = id(item)
                if obj_id in visited:
                    items.append(f"{indent_str}  - : <circular reference>")
                else:
                    visited.add(obj_id)
                    item_str = item.__str__(indent + 2, visited, depth + 1).strip()
                    visited.discard(obj_id)
                    items.append(f"{indent_str}  - : {item_str}" if item_str else f"{indent_str}  - :")
            else:
                items.append(f"{indent_str}  - : {item}" if item != "" else f"{indent_str}  - :")
        return f"{indent_str}{key}:\n" + "\n".join(items)

    def _format_dict_field(self, key: str, value: dict, indent: int, visited: set, depth: int) -> str:
        """Format a dictionary field."""
        indent_str = " " * indent
        if not value:
            return f"{indent_str}{key}:\n"

        # Check depth limit first
        if depth >= self._MAX_RECURSION_DEPTH:
            return f"{indent_str}{key}: <max depth reached>"

        # Check for circular reference
        obj_id = id(value)
        if obj_id in visited:
            return f"{indent_str}{key}: <circular reference>"

        visited.add(obj_id)
        items = []
        for k, v in value.items():
            if isinstance(v, (dict, list, LLMDataModel)):
                nested_str = self.format_field(k, v, indent + 2, visited, depth + 1)
                items.append(nested_str)
            else:
                items.append(f"{indent_str}  {k}: {v}")
        visited.discard(obj_id)
        return f"{indent_str}{key}:\n" + "\n".join(items) if key else "\n".join(items)

    def _format_primitive_field(self, key: str, value: Any, indent: int, _visited: set, _depth: int) -> str:
        """Format a primitive field."""
        return f"{' ' * indent}{key}: {value}"

    def __str__(self, indent: int = 0, visited: set | None = None, depth: int = 0) -> str:
        """
        Converts the model into a formatted string for LLM prompts.
        Handles indentation for nested models and includes an optional section header.
        """
        if visited is None:
            visited = set()
        indent_str = " " * indent
        field_list = [
            self.format_field(name, getattr(self, name), indent + 2, visited, depth)
            for name, field in type(self).model_fields.items()
            if (
                not getattr(field, "exclude", False)
                and name != "section_header"
            )
        ]

        fields = "\n".join(field_list) + "\n" if field_list else ""

        if self.section_header and indent == 0:
            header = f"{indent_str}[[{self.section_header}]]\n"
            return f"{header}{fields}"
        return fields

    def validate(self) -> bool:
        """Custom validation of the model."""
        return None

    def remedy(self):
        """Default remedy method for the model."""
        return

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

        main_schema = cls._format_schema_fields(properties, schema, definitions, 0)
        definitions_schema = cls._format_schema_definitions(definitions, schema)

        return cls._compose_schema_output(main_schema, definitions_schema)

    @classmethod
    def _extract_schema_properties(cls, schema: dict) -> dict:
        """Extract properties from schema."""
        # Direct properties available
        if "properties" in schema and isinstance(schema["properties"], dict):
            return schema["properties"]

        # Look into referenced root definitions
        defs = cls._extract_schema_definitions(schema)
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            ref = defs.get(ref_name, {})
            return ref.get("properties", {})

        # Merge from allOf if used at root
        if "allOf" in schema and isinstance(schema["allOf"], list):
            merged = {}
            for part in schema["allOf"]:
                if "$ref" in part:
                    ref_name = part["$ref"].split("/")[-1]
                    ref = defs.get(ref_name, {})
                    merged.update(ref.get("properties", {}))
                elif "properties" in part and isinstance(part["properties"], dict):
                    merged.update(part["properties"])
            if merged:
                return merged
        return {}

    @classmethod
    def _extract_schema_definitions(cls, schema: dict) -> dict:
        """Extract definitions from schema."""
        return schema.get("$defs", schema.get("definitions", {}))

    @classmethod
    def _format_schema_field(cls, name: str, field_schema: dict, required: bool,
                            definitions: dict, indent_level: int, visited: set | None = None) -> str:
        """Format a single schema field without descriptions (kept for definitions)."""
        visited = visited or set()

        field_type = cls._resolve_field_type(field_schema, definitions)

        is_required = "required" if required else "optional"
        indent = "  " * indent_level

        nested_description = ""
        if field_type.startswith("nested object"):
            ref_name = field_schema.get("$ref", "").split("/")[-1]
            nested_description = cls._format_referenced_object_fields(
                ref_name, definitions, indent_level, visited
            )
        elif field_type.startswith("array of nested object"):
            nested_description = cls._format_array_referenced_object_fields(
                field_schema, definitions, indent_level, visited
            )

        # Include const note (no descriptions in schema section)
        const_note = ""
        if "const_value" in field_schema:
            const_note = f' [const: "{field_schema["const_value"]}"]'
        result = f'{indent}- "{name}" ({field_type}, {is_required}){const_note}'
        if nested_description:
            result += f"\n{indent}  - Nested fields:\n{nested_description}"
        return result

    @classmethod
    def _format_referenced_object_fields(cls, ref_name: str, definitions: dict,
                                        indent_level: int, visited: set) -> str:
        """Format nested fields for a referenced object definition by name."""
        if ref_name in definitions and ref_name not in visited:
            visited.add(ref_name)
            return cls._format_schema_fields(
                definitions[ref_name].get("properties", {}),
                definitions[ref_name], definitions, indent_level + 1, visited.copy()
            )
        return ""

    @classmethod
    def _format_array_referenced_object_fields(cls, field_schema: dict, definitions: dict,
                                              indent_level: int, visited: set) -> str:
        """Format nested fields for arrays referencing object definitions."""
        ref_name = field_schema.get("items", {}).get("$ref", "").split("/")[-1]
        return cls._format_referenced_object_fields(ref_name, definitions, indent_level, visited)

    @classmethod
    def _format_schema_fields(cls, properties: dict, schema: dict, definitions: dict,
                             indent_level: int, visited: set | None = None) -> str:
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

        return "\n".join(lines)

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
        """Resolve allOf schema to a string type description."""
        parts = field_schema.get("allOf", [])
        if not isinstance(parts, list) or not parts:
            return "unknown"
        # If a single $ref, resolve to nested object name
        if len(parts) == 1 and "$ref" in parts[0]:
            ref_name = parts[0]["$ref"].split("/")[-1]
            if ref_name in definitions and isinstance(definitions[ref_name], dict):
                ref_def = definitions[ref_name]
                if "enum" in ref_def:
                    return cls._format_enum_type(ref_def["enum"])
            return f"nested object ({ref_name})"
        # Multiple parts: treat as composite object
        return "object"

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
        # Check if it's a set (has uniqueItems: true)
        if field_schema.get("uniqueItems") is True:
            items = field_schema.get("items", {})
            item_type = cls._resolve_field_type(items, definitions)
            return f"set of {item_type}"

        # Check if it's a tuple (has prefixItems)
        if "prefixItems" in field_schema:
            prefix_items = field_schema["prefixItems"]
            item_types = [cls._resolve_field_type(item, definitions) for item in prefix_items]
            return f"tuple of ({', '.join(item_types)})"

        # Regular list/array
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
    def _format_schema_definitions(cls, definitions: dict, root_schema: dict | None = None) -> str:
        """Format schema definitions using descriptions and examples; omit redundant types.

        Also includes the root model's fields (from root_schema) so their descriptions/examples
        are visible, not just $defs.
        """
        lines: list[str] = []
        visited_defs: set[str] = set()

        lines.extend(cls._format_root_definition_lines(root_schema))

        for name, definition in definitions.items():
            if name in visited_defs:
                continue
            visited_defs.add(name)
            lines.extend(cls._format_single_definition(name, definition))

        return "\n".join(lines)

    @classmethod
    def _format_root_definition_lines(cls, root_schema: dict | None) -> list[str]:
        """Format definitions derived from the root schema."""
        if not (root_schema and isinstance(root_schema, dict)):
            return []
        root_props = cls._extract_schema_properties(root_schema)
        if not root_props:
            return []
        root_title = root_schema.get("title", "Root")
        lines = [f"- {root_title}:"]
        lines.extend(cls._format_definition_properties(root_props))
        return lines

    @classmethod
    def _format_single_definition(cls, name: str, definition: dict) -> list[str]:
        """Format a single definition block, including enum handling."""
        lines = [f"- {name}:"]
        if "enum" in definition:
            enum_values = ", ".join(map(repr, definition["enum"]))
            lines.append(f"  Enum values: {enum_values}")
            return lines
        props = definition.get("properties", {})
        lines.extend(cls._format_definition_properties(props))
        return lines

    @classmethod
    def _format_definition_properties(cls, props: dict) -> list[str]:
        """Render property lines using descriptions and examples."""
        out: list[str] = []
        for prop_name, prop_schema in props.items():
            if prop_name == "section_header":
                continue
            out.append(cls._format_property_description(prop_name, prop_schema))
            out.extend(cls._format_property_examples(prop_schema))
        return out

    @staticmethod
    def _format_property_description(prop_name: str, prop_schema: dict) -> str:
        """Format the description line for a property, including const hints."""
        desc = prop_schema.get("description")
        const_note = ""
        if "const_value" in prop_schema:
            const_note = f' (const value: "{prop_schema["const_value"]}")'
        if not desc:
            return (
                f'  - "{prop_name}": '
                "No definition provided. Focus on the [[Schema]] and the prompt to infer "
                "the expected structure and constraints."
            )
        return f'  - "{prop_name}": {desc}{const_note}'

    @classmethod
    def _format_property_examples(cls, prop_schema: dict) -> list[str]:
        """Format example lines for a property schema."""
        examples = prop_schema.get("examples")
        if examples is None and "example" in prop_schema:
            examples = prop_schema.get("example")
        if isinstance(examples, (list, tuple)):
            if not examples:
                return []
            lines = ["    - Examples:"]
            for example in examples:
                lines.append(f"      - {cls._format_example_value(example)}")
            return lines
        if examples is not None:
            return [f"    - Example: {cls._format_example_value(examples)}"]
        return []

    @staticmethod
    def _format_example_value(val: Any) -> str:
        """Safely format example values for display, preserving human readability."""
        if isinstance(val, str):
            return val
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)

    @classmethod
    def _generate_type_description(cls, type_desc: str | dict) -> str:
        """Generate a human-readable description for a type."""
        normalized = cls._normalize_type_descriptor(type_desc)
        if normalized is None:
            return "unknown"

        composite_description = cls._describe_composite_type(normalized)
        return composite_description if composite_description is not None else normalized

    @classmethod
    def _normalize_type_descriptor(cls, type_desc: Any) -> str | None:
        """Normalize a type descriptor into a descriptive string."""
        if type_desc is None:
            return None
        if isinstance(type_desc, dict):
            type_desc = cls._resolve_field_type(type_desc, {})
        if isinstance(type_desc, type):
            type_desc = type_desc.__name__
        if not isinstance(type_desc, str):
            type_desc = str(type_desc)
        return type_desc

    @classmethod
    def _describe_composite_type(cls, type_desc: str) -> str | None:
        """Describe composite collection-like types with friendly language."""
        handlers = {
            "array of ": cls._describe_list_type,
            "set of ": cls._describe_set_type,
            "tuple of ": cls._describe_tuple_type,
            "object of ": cls._describe_dict_type,
        }
        for prefix, handler in handlers.items():
            if type_desc.startswith(prefix):
                item_type = type_desc[len(prefix):]
                return handler(item_type)
        return None

    @classmethod
    def _describe_list_type(cls, item_type: str) -> str:
        """Describe a list-style type."""
        element_desc = cls._nested_element_description(item_type)
        if element_desc is None:
            return f"A list containing {item_type} values"
        return f"A list where each element is {element_desc}"

    @classmethod
    def _describe_set_type(cls, item_type: str) -> str:
        """Describe a set-style type."""
        element_desc = cls._nested_element_description(item_type)
        if element_desc is None:
            return f"A set containing unique {item_type} values"
        return f"A set where each element is {element_desc}"

    @staticmethod
    def _describe_tuple_type(item_type: str) -> str:
        """Describe a tuple-style type."""
        return f"A tuple with specific types: {item_type}"

    @classmethod
    def _describe_dict_type(cls, item_type: str) -> str:
        """Describe a dictionary-style type."""
        element_desc = cls._nested_element_description(item_type)
        if element_desc is None:
            return f"A dictionary with {item_type} values"
        return f"A dictionary where each value is {element_desc}"

    @staticmethod
    def _nested_element_description(item_type: str) -> str | None:
        """Convert nested composite descriptors into human-friendly text."""
        if item_type.startswith("nested object"):
            return item_type
        nested_mappings = {
            "array of ": "a list of {} values",
            "set of ": "a set of unique {} values",
            "tuple of ": "a tuple with types: {}",
            "object of ": "a dictionary with {} values",
        }
        for prefix, template in nested_mappings.items():
            if item_type.startswith(prefix):
                inner = item_type[len(prefix):]
                return template.format(inner)
        return None

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
    def _generate_example_for_model(model: type[BaseModel], visited_models: set) -> dict:
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
        # Honor const fields explicitly
        if LLMDataModel._is_const_field(model_field):
            return LLMDataModel._get_const_value(model_field)

        if LLMDataModel._has_default_value(model_field):
            default_val = model_field.default
            desc = getattr(model_field, 'description', None)
            ann = getattr(model_field, 'annotation', None)
            is_desc_like = isinstance(default_val, str) and (
                (desc and default_val.strip() == str(desc).strip()) or
                len(default_val) >= 30 or
                any(kw in default_val for kw in ["represents", "should", "Always use", "This is", "This represents"])
            )
            if is_desc_like and (ann is str or ann is Any or ann is None):
                return "example_string"
            return default_val
        if model_field.default_factory is not None:
            # For example generation, we want to show structure even if default is empty
            # Check if default_factory would produce an empty container
            default_val = model_field.default_factory()
            if isinstance(default_val, (list, dict, set, tuple)) and len(default_val) == 0:
                # Generate example data instead of using empty default
                return LLMDataModel._generate_value_for_type(
                    model_field.annotation, visited_models
                )
            return default_val
        return LLMDataModel._generate_value_for_type(
            model_field.annotation, visited_models
        )

    @staticmethod
    def _generate_value_for_type(field_type: Any, visited_models: set) -> Any:
        """Generate a value for a specific type (standard behavior)."""
        return LLMDataModel._generate_value_for_type_generic(field_type, visited_models, prefer_non_null=False)

    @staticmethod
    def _generate_union_value(field_type: Any, visited_models: set) -> Any:
        """Generate a value for a Union type."""
        subtypes = LLMDataModel._get_union_types(field_type, exclude_none=True)
        if not subtypes:
            return None
        return LLMDataModel._generate_value_for_type_generic(subtypes[0], visited_models, prefer_non_null=False)

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
            return [LLMDataModel._generate_value_for_type_generic(subtype, visited_models, False) for subtype in subtypes[:2]]

        return [LLMDataModel._generate_value_for_type_generic(item_type, visited_models, False)]

    @staticmethod
    def _generate_dict_value(field_type: Any, visited_models: set) -> dict:
        """Generate a value for a dict type."""
        key_type, value_type = get_args(field_type) if get_args(field_type) else (Any, Any)

        example_key = LLMDataModel._example_key_for_type(key_type, visited_models)
        return {example_key: LLMDataModel._generate_value_for_type_generic(value_type, visited_models, False)}

    @staticmethod
    def _generate_set_value(field_type: Any, visited_models: set) -> list:
        """Generate a value for a set type (returns list for JSON serialization)."""
        item_type = get_args(field_type)[0] if get_args(field_type) else Any
        return [LLMDataModel._generate_value_for_type_generic(item_type, visited_models, False)]

    @staticmethod
    def _generate_tuple_value(field_type: Any, visited_models: set) -> tuple:
        """Generate a value for a tuple type."""
        types = get_args(field_type)
        if types:
            return tuple(LLMDataModel._generate_value_for_type_generic(t, visited_models, False) for t in types)
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
            # Prefer non-null examples for optionals in instruct examples
            examples = [cls._generate_single_example_non_null()]

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
            examples.append(cls._format_json_example(example))
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
            examples.append(cls._format_json_example({field_name: value_example}))
        return examples

    @classmethod
    def _generate_single_example_non_null(cls):
        """Generate a single example for the model preferring non-null optionals."""
        example = cls._generate_non_null_example_for_model(cls)
        return cls._format_json_example(example)

    @classmethod
    def _create_example_with_type(cls, subtype: Any) -> dict:
        """Create an example dict for a specific type."""
        submodel = build_dynamic_llm_datamodel(subtype)
        return submodel.generate_example_json()

    @classmethod
    def _generate_non_null_example_for_model(cls, model: type[BaseModel], visited_models: set | None = None) -> dict:
        """Generate an example for a model, preferring non-null for Optional fields (recursive)."""
        if visited_models is None:
            visited_models = set()
        example = {}
        for field_name, model_field in model.model_fields.items():
            if field_name == "section_header":
                continue
            # Const takes precedence
            if LLMDataModel._is_const_field(model_field):
                example[field_name] = LLMDataModel._get_const_value(model_field)
                continue

            annotation = model_field.annotation
            if LLMDataModel._is_optional_type(annotation):
                non_none_types = LLMDataModel._get_union_types(annotation, exclude_none=True)
                chosen = non_none_types[0] if non_none_types else Any
                example[field_name] = cls._generate_value_for_type_non_null(chosen, visited_models)
            else:
                example[field_name] = cls._generate_value_for_type_non_null(model_field.annotation, visited_models)
        return example

    @classmethod
    def _generate_value_for_type_non_null(cls, field_type: Any, visited_models: set) -> Any:
        """Generate a value ensuring non-null choices for unions/optionals."""
        return cls._generate_value_for_type_generic(field_type, visited_models, prefer_non_null=True)

    @classmethod
    def _example_key_for_type(cls, key_type: Any, visited_models: set) -> Any:
        """Generate an example key for dicts based on key type."""
        if key_type is int:
            return 123
        if key_type is float:
            return 123.45
        if key_type is bool:
            return True
        if key_type is tuple or get_origin(key_type) is tuple:
            tuple_args = get_args(key_type) if get_args(key_type) else (str, int)
            return tuple(cls._generate_value_for_type_generic(t, visited_models, True) for t in tuple_args)
        if key_type is frozenset or get_origin(key_type) is frozenset:
            item_type = get_args(key_type)[0] if get_args(key_type) else str
            return frozenset([cls._generate_value_for_type_generic(item_type, visited_models, True)])
        return "example_string"

    @classmethod
    def _generate_value_for_type_generic(cls, field_type: Any, visited_models: set, prefer_non_null: bool) -> Any:
        """Unified generator for example values; prefer_non_null to avoid None variants."""
        origin = get_origin(field_type) or field_type

        handled, value = cls._handle_enum_type(origin)
        if handled:
            return value

        handled, value = cls._handle_literal_type(origin, field_type)
        if handled:
            return value

        handled, value = cls._handle_model_type(origin, field_type, visited_models, prefer_non_null)
        if handled:
            return value

        handled, value = cls._handle_union_type(field_type, visited_models, prefer_non_null)
        if handled:
            return value

        handled, value = cls._handle_collection_type(field_type, visited_models, prefer_non_null)
        if handled:
            return value

        return LLMDataModel._generate_primitive_value(field_type)

    @staticmethod
    def _handle_enum_type(origin: Any) -> tuple[bool, Any]:
        """Handle Enum types when generating example values."""
        if isinstance(origin, type) and issubclass(origin, Enum):
            first_member = next(iter(origin), None)
            value = first_member.value if first_member is not None else "enum_value"
            return True, value
        return False, None

    @staticmethod
    def _handle_literal_type(origin: Any, field_type: Any) -> tuple[bool, Any]:
        """Handle Literal[...] annotations."""
        if origin is Literal:
            literal_args = get_args(field_type)
            return True, literal_args[0] if literal_args else None
        return False, None

    @classmethod
    def _handle_model_type(cls, origin: Any, field_type: Any, visited_models: set, prefer_non_null: bool) -> tuple[bool, Any]:
        """Handle Pydantic BaseModel subclasses."""
        if not (isinstance(origin, type) and issubclass(origin, BaseModel)):
            return False, None
        model_name = field_type.__name__
        if model_name in visited_models:
            return True, {}
        visited_models.add(model_name)
        generator = cls._generate_non_null_example_for_model if prefer_non_null else LLMDataModel._generate_example_for_model
        return True, generator(field_type, visited_models.copy())

    @classmethod
    def _handle_union_type(cls, field_type: Any, visited_models: set, prefer_non_null: bool) -> tuple[bool, Any]:
        """Handle Optional/Union annotations."""
        if not LLMDataModel._is_union_type(field_type):
            return False, None
        subtypes = LLMDataModel._get_union_types(field_type, exclude_none=True)
        if not subtypes:
            return True, None
        chosen = subtypes[0]
        value = cls._generate_value_for_type_generic(chosen, visited_models, prefer_non_null)
        return True, value

    @classmethod
    def _handle_collection_type(cls, field_type: Any, visited_models: set, prefer_non_null: bool) -> tuple[bool, Any]:
        """Handle list/dict/set/tuple-like annotations."""
        if not LLMDataModel._is_collection_type(field_type):
            return False, None

        origin = get_origin(field_type) or field_type
        args = get_args(field_type)

        if origin is list:
            item_type = args[0] if args else Any
            value = [cls._generate_value_for_type_generic(item_type, visited_models, prefer_non_null)]
            return True, value
        if origin is dict:
            key_type, value_type = args if args else (Any, Any)
            example_key = cls._example_key_for_type(key_type, visited_models)
            example_value = cls._generate_value_for_type_generic(value_type, visited_models, prefer_non_null)
            return True, {example_key: example_value}
        if origin in (set, frozenset):
            item_type = args[0] if args else Any
            value = [cls._generate_value_for_type_generic(item_type, visited_models, prefer_non_null)]
            return True, value
        if origin is tuple:
            if args:
                tuple_values = tuple(
                    cls._generate_value_for_type_generic(t, visited_models, prefer_non_null)
                    for t in args
                )
                return True, tuple_values
            return True, ("item1", "item2")

        return True, []

    @classmethod
    def _format_json_example(cls, obj: Any, _indent: int = 0) -> str:
        """Format an object as a JSON string representation."""
        return json.dumps(obj, indent=2, default=str)

    @classmethod
    def _format_examples(cls, examples: list) -> str:
        """Format examples into the final output string."""
        if len(examples) == 1:
            return f"[[Example]]\n```json\n{examples[0]}\n```"

        example_blocks = []
        for idx, ex in enumerate(examples, start=1):
            example_blocks.append(f"[[Example {idx}]]\n```json\n{ex}\n```")
        return "\n\n".join(example_blocks)


def build_dynamic_llm_datamodel(py_type: Any) -> type[LLMDataModel]:
    """Dynamically create a subclass of LLMDataModel with a single 'value' field."""
    model_name = f"LLMDynamicDataModel_{hash(str(py_type)) & 0xFFFFFFFF:X}"

    model: type[LLMDataModel] = create_model(
        model_name,
        __base__=LLMDataModel,
        value=(
            py_type,
            Field(
                ...,
                description="This is a dynamically generated data model. This description is general. "
                "If you're dealing with a complex type, or nested types in combination with unions, make sure you "
                "understand the instructions provided in the prompt, and select the appropriate data model based on the "
                "type at hand, as described in the schema section."
            )
        ),
    )

    return model
