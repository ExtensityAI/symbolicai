from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import TYPE_CHECKING, Any, Literal, Union, cast, get_args, get_origin, override

from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_core import PydanticUndefined

if TYPE_CHECKING:
    from pydantic import GetJsonSchemaHandler
    from pydantic.fields import FieldInfo
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema


def Const(value: str) -> Any:
    """Declare a string field whose value is fixed."""
    return Field(default=value, json_schema_extra={"const": value})


class LLMDataModel(BaseModel):
    """Pydantic model with prompt rendering and structured-output instructions."""

    section_header: str | None = Field(default=None, exclude=True, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def validate_const_fields(cls, values: object) -> object:
        if not isinstance(values, Mapping):
            return values
        for name, field in cls.model_fields.items():
            expected = _const_value(field)
            if expected is not PydanticUndefined and name in values and values[name] != expected:
                msg = f"{name} must be {expected!r}"
                raise ValueError(msg)
        return values

    @classmethod
    @override
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("section_header", None)
        required = schema.get("required")
        if isinstance(required, list) and "section_header" in required:
            required.remove("section_header")
        return schema

    def render(self) -> str:
        """Render this value as compact, readable prompt context."""
        lines = []
        if self.section_header:
            lines.append(f"[[{self.section_header}]]")
        for name, field in type(self).model_fields.items():
            if name == "section_header" or field.exclude:
                continue
            lines.extend(_render_field(name, getattr(self, name), 0))
        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.render()

    @classmethod
    @lru_cache(maxsize=128)
    def simplify_json_schema(cls) -> str:
        """Return the model schema in an LLM-readable JSON block."""
        schema = json.dumps(cls.model_json_schema(), indent=2, ensure_ascii=False)
        return f"[[Schema]]\n```json\n{schema}\n```"

    @classmethod
    @lru_cache(maxsize=128)
    def instruct_llm(cls) -> str:
        """Describe the required JSON result and provide one valid example."""
        example = json.dumps(cls.generate_example_json(), indent=2, ensure_ascii=False)
        return (
            "[[Result]]\nReturn a JSON object matching this schema.\n\n"
            f"{cls.simplify_json_schema()}\n\n"
            f"[[Example]]\n```json\n{example}\n```"
        )

    @classmethod
    def generate_example_json(cls) -> dict[str, object]:
        """Generate one deterministic value accepted by this model."""
        return _example_for_model(cls, set())


class _DynamicLLMDataModel(LLMDataModel):
    value: Any


def build_dynamic_llm_datamodel(py_type: Any) -> type[_DynamicLLMDataModel]:
    """Wrap a Python type in a single-field LLM data model."""
    model_name = f"LLMDynamicDataModel_{hash(str(py_type)) & 0xFFFFFFFF:X}"
    return cast(
        "type[_DynamicLLMDataModel]",
        create_model(
            model_name,
            __base__=_DynamicLLMDataModel,
            value=(
                py_type,
                Field(..., description="The typed value returned by the language model."),
            ),
        ),
    )


def _const_value(field: FieldInfo) -> object:
    extra = field.json_schema_extra
    if isinstance(extra, dict) and "const" in extra:
        return extra["const"]
    return PydanticUndefined


def _render_field(name: str, value: object, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, LLMDataModel):
        lines = [f"{prefix}{name}:"]
        for child_name, child_field in type(value).model_fields.items():
            if child_name == "section_header" or child_field.exclude:
                continue
            lines.extend(_render_field(child_name, getattr(value, child_name), indent + 2))
        return lines
    if isinstance(value, list):
        lines = [f"{prefix}{name}:"]
        for item in value:
            if isinstance(item, LLMDataModel):
                rendered = item.render().splitlines()
                lines.append(f"{' ' * (indent + 2)}- {rendered[0]}")
                lines.extend(f"{' ' * (indent + 4)}{line}" for line in rendered[1:])
            else:
                lines.append(f"{' ' * (indent + 2)}- {item}")
        return lines
    if isinstance(value, dict):
        lines = [f"{prefix}{name}:"]
        for key, item in value.items():
            lines.extend(_render_field(str(key), item, indent + 2))
        return lines
    if isinstance(value, Enum):
        value = value.value
    return [f"{prefix}{name}: {value}"]


def _example_for_model(model: type[BaseModel], seen: set[type[BaseModel]]) -> dict[str, object]:
    if model in seen:
        return {}
    nested_seen = {*seen, model}
    example: dict[str, object] = {}
    for name, field in model.model_fields.items():
        if name == "section_header" or field.exclude:
            continue
        const = _const_value(field)
        if const is not PydanticUndefined:
            example[name] = const
            continue
        if field.default is not PydanticUndefined:
            example[name] = field.default
            continue
        if field.default_factory is not None:
            example[name] = field.get_default(call_default_factory=True, validated_data={})
            continue
        example[name] = _example_for_type(field.annotation, nested_seen)
    return example


def _example_for_type(annotation: object, seen: set[type[BaseModel]]) -> object:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        choices = [choice for choice in args if choice is not type(None)]
        return _example_for_type(choices[0], seen) if choices else None
    if origin is Literal:
        return args[0] if args else None
    if origin is list:
        return [_example_for_type(args[0] if args else Any, seen)]
    if origin in (set, frozenset):
        return [_example_for_type(args[0] if args else Any, seen)]
    if origin is tuple:
        return [_example_for_type(item, seen) for item in args if item is not Ellipsis]
    if origin is dict:
        value_type = args[1] if len(args) == 2 else Any
        return {"key": _example_for_type(value_type, seen)}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _example_for_model(annotation, seen)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        member = next(iter(annotation), None)
        return member.value if member is not None else None
    if annotation is str:
        return "example"
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    if annotation is bool:
        return True
    return "example"
