from collections.abc import Sequence
from typing import Self

from pydantic import ValidationError, model_validator

from symai.contract.models import LLMDataModel
from symai.operations import language_request
from symai.runtime.models import JsonSchemaResponseFormat, LanguageModelRequest
from symai.runtime.runtime import LanguageModel


class _SemanticVerdict(LLMDataModel):
    valid: bool
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.valid and self.errors:
            msg = "A valid semantic verdict cannot contain errors"
            raise ValueError(msg)
        if not self.valid and not self.errors:
            msg = "An invalid semantic verdict must explain at least one error"
            raise ValueError(msg)
        return self


def parse_output[OutputT: LLMDataModel](
    output_text: str,
    output_type: type[OutputT],
) -> OutputT:
    """Parse model text exactly once into the contract's output type."""
    return output_type.model_validate_json(output_text)


def validation_errors(error: ValidationError) -> tuple[str, ...]:
    """Render Pydantic failures as stable, prompt-safe field messages."""
    rendered = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        location = " -> ".join(str(part) for part in detail["loc"]) or "value"
        rendered.append(f"{location}: {detail['msg']}")
    return tuple(rendered)


def build_remedy_prompt(
    *,
    instruction: str,
    input_text: str,
    output_type: type[LLMDataModel],
    output_text: str,
    errors: Sequence[str],
) -> str:
    """Build the bounded correction request for one failed output."""
    error_text = "\n".join(f"- {error}" for error in errors)
    return (
        "Correct a language-model result that violated a typed contract.\n\n"
        f"[[Instruction]]\n{instruction}\n\n"
        f"[[Input]]\n{input_text}\n\n"
        f"{output_type.instruct_llm()}\n\n"
        f"[[Invalid Output]]\n{output_text}\n\n"
        f"[[Validation Errors]]\n{error_text}\n\n"
        "Preserve valid information and change only what is needed to resolve every error. "
        "Return only the corrected JSON object."
    )


def structured_request(
    instruction: str,
    value: object,
    output_type: type[LLMDataModel],
) -> LanguageModelRequest:
    """Construct one provider-neutral JSON Schema request."""
    request = language_request(instruction, str(value))
    response_format = JsonSchemaResponseFormat(
        name=output_type.__name__,
        json_schema=output_type.model_json_schema(),
        strict=True,
    )
    return request.model_copy(update={"response_format": response_format})


def check_semantic_conditions(
    engine: LanguageModel,
    output: LLMDataModel,
    conditions: Sequence[str],
) -> tuple[str, ...]:
    """Judge natural-language postconditions through an explicit model handle."""
    if not conditions:
        return ()
    condition_text = "\n".join(f"- {condition}" for condition in conditions)
    instruction = (
        "Judge whether the typed output satisfies every semantic condition. "
        "Do not repair the output. Return valid=true with no errors, or valid=false "
        "with one concise error per violated condition.\n\n"
        f"[[Conditions]]\n{condition_text}"
    )
    response = engine.execute(structured_request(instruction, output.render(), _SemanticVerdict))
    try:
        verdict = parse_output(response.outputs[0].text, _SemanticVerdict)
    except ValidationError as error:
        return tuple(f"semantic judge: {message}" for message in validation_errors(error))
    return () if verdict.valid else verdict.errors
