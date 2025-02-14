import logging
from pydoc import locate
from typing import Type

import numpy as np
from loguru import logger
from pydantic import BaseModel, ValidationError

from .core_ext import deprecated
from .components import Function
from .models import (ExceptionWithUsage, LengthConstraint, LLMDataModel,
                     TypeValidationError)
from .symbol import Expression

NUM_REMEDY_RETRIES = 10

class ValidatedFunction(Function):
    _default_retry_params = dict(
        tries=5,
        delay=0.5,
        max_delay=15,
        backoff=2,
        jitter=0.1,
        graceful=False
    )
    def __init__(
        self,
        data_model: LLMDataModel = None,
        retry_params: dict[str, int | float | bool] = _default_retry_params,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retry_params = retry_params
        self.data_model = data_model
        self.remedy_function = Function(
            """
            [Task]
            Fix the provided JSON string to ensure it is valid according to the schema and resolves all listed validation errors.

            [Important Guidelines]
            1. Only address the specific issues described in the validation errors.
            2. Preserve the meaning and values of the original JSON as much as possible unless changes are necessary for schema compliance.
            3. Ensure that the corrected JSON is both well-formatted and valid for the given schema.
            4. Return the corrected JSON string as the output.
            """,
            static_context="""
            You are tasked with fixing a string that is intended to be in **JSON format** but contains errors.
            Your goal is to correct the errors and ensure the JSON string is valid according to a given JSON schema.
            Follow these rules:

            1. Parse the provided string and use the list of validation errors to identify what needs to be fixed.
            2. Correct the identified errors to produce a properly formatted JSON string.
            3. Ensure the corrected JSON complies fully with the provided JSON schema.
            4. Preserve all original keys and values as much as possible. Only modify keys or values if they do not comply with the schema.
            5. Only modify the structure or values if necessary to meet the schema's requirements.
            6. Return the corrected JSON string as the output.

            [Requirements]
            - The output must be a valid, well-formatted JSON string.
            - Do not introduce new data or alter the intent of the original content unless required for schema compliance.
            - Ensure all changes are minimal and strictly necessary to fix the listed errors.
            """,
            response_format={"type": "json_object"},
        )

    def prepare_seeds(self, num_seeds: int, **kwargs):
        # get list of seeds for remedy (to avoid same remedy for same input)
        if "seed" in kwargs:
            seed = kwargs["seed"]
        elif hasattr(self, "seed"):
            seed = self.seed
        else:
            seed = 42

        rnd = np.random.RandomState(seed=seed)
        seeds = rnd.randint(
            0, np.iinfo(np.int16).max, size=num_seeds, dtype=np.int16
        ).tolist()
        return seeds

    def simplify_validation_errors(self, error: ValidationError) -> str:
        """
        Simplifies Pydantic validation errors into a concise, LLM-friendly format, including lists and nested elements.

        Args:
            error (ValidationError): The Pydantic ValidationError instance.

        Returns:
            str: A simplified and actionable error message.
        """
        simplified_errors = []
        for err in error.errors():
            # Build a human-readable field path
            field_path = " -> ".join(
                [str(element) for element in err["loc"]]
            )  # Includes indices for lists, keys, etc.
            message = err["msg"]  # Error message
            expected_type = err.get("type", "unknown")  # Expected type (if available)
            provided_value = err.get("ctx", {}).get(
                "given", "unknown"
            )  # Provided value (if available)

            # Create a concise, actionable error message
            error_message = (
                f"Field '{field_path}': {message}. "
                f"Expected type: {expected_type}. Provided value: {provided_value}."
            )
            simplified_errors.append(error_message)

        # Combine all errors into a single message
        return "\n".join(simplified_errors)

    def forward(self, *args, **kwargs):
        # Force JSON mode
        kwargs["response_format"] = {"type": "json_object"}
        json = super().forward(*args, **kwargs)
        # Get list of seeds for remedy (to avoid same remedy for same input)
        remedy_seeds = self.prepare_seeds(self.retry_params["tries"], **kwargs)

        result = None
        last_error = ""
        for i in range(self.retry_params["tries"]):
            try:
                # Try to validate against provided data model
                result = self.data_model.model_validate_json(json, strict=True)
                break
            except ValidationError as e:
                self._pause()

                error_str = self.simplify_validation_errors(e)

                # Change the dynamic context of the remedy function to account for the errors
                self.remedy_function.clear()
                self.remedy_function.adapt(f"[Original Input]\n```json\n{json}\n´´´\n")
                self.remedy_function.adapt(f"[Validation Errors]\n{error_str}\n")
                self.remedy_function.adapt(f"[JSON Schema]\n{self.data_model.instruct_llm()}\n")
                json = self.remedy_function(seed=remedy_seeds[i]).value

                last_error = error_str

        if result is None:
            raise TypeValidationError(f"Failed to retrieve valid JSON: {last_error}")

        return result

    def _pause(self):
        _delay = self.retry_params['delay']
        _delay *= self.retry_params['backoff']

        if isinstance(self.retry_params['jitter'], tuple):
            _delay += np.random.uniform(*self.retry_params['jitter'])
        else:
            _delay += self.retry_params['jitter']

        if self.retry_params['max_delay'] >= 0:
            _delay = min(_delay, self.retry_params['max_delay'])


#TODO: this is still dependent on the old usage; refactor
@deprecated("This will be changed in upcoming releases to not depend on the old usage implementation. For usage, see `symai.components.MetadataTracker`.")
class LengthConstrainedFunction(ValidatedFunction):
    def __init__(
        self,
        character_constraints: list[LengthConstraint] | LengthConstraint,
        constraint_retry_count: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.character_constraints = character_constraints
        self.constraint_retry_count = constraint_retry_count

    def forward(self, *args, **kwargs):
        result, usage = super().forward(*args, **kwargs)

        original_task = args[0]  # TODO can we validate this?

        # get list of seeds for remedy (to avoid same remedy for same input)
        remedy_seeds = self.prepare_seeds(self.constraint_retry_count, **kwargs)
        for i in range(self.constraint_retry_count):
            constraint_violations = self.check_constraints(result)
            if len(constraint_violations) > 0:
                self.print_verbose(f"Constraint violations: {constraint_violations}")
                remedy_task = self.wrap_task(
                    original_task, result.model_dump_json(), constraint_violations
                )
                kwargs["seed"] = remedy_seeds[i]
                result, remedy_usage = super().forward(remedy_task, *args[1:], **kwargs)
                # update local usage
                usage.prompt_tokens += remedy_usage.prompt_tokens
                usage.completion_tokens += remedy_usage.completion_tokens
                usage.total_tokens += remedy_usage.total_tokens
            else:
                break

        last_violation = self.check_constraints(result)
        if i == self.constraint_retry_count and len(last_violation) > 0:
            raise ExceptionWithUsage(
                f"Failed to enforce length constraints: {' | '.join(last_violation)}", usage
            )

        return result, usage

    def check_constraints(self, result: BaseModel):
        constraint_violations = []
        if type(self.character_constraints) is not list:
            character_constraints = [self.character_constraints]
        else:
            character_constraints = self.character_constraints

        for constraint in character_constraints:
            for field_value in self.get(result, constraint.field_name):
                if (
                    not constraint.min_characters
                    <= len(field_value)
                    <= constraint.max_characters
                ):
                    # TODO improve for lists (especially table of contents) by providing the index
                    self.print_verbose(
                        f"Field {constraint.field_name} must have between {constraint.min_characters} and {constraint.max_characters} characters, has {len(field_value)}"
                    )
                    remedy_str = [
                        f"The field {constraint.field_name} must have between {constraint.min_characters} and {constraint.max_characters} characters, but has {len(field_value)}."
                    ]
                    if len(field_value) < constraint.min_characters:
                        remedy_str.append(
                            f"Increase the length of {constraint.field_name} by at least {constraint.min_characters - len(field_value)} characters."
                        )
                    elif len(field_value) > constraint.max_characters:
                        remedy_str.append(
                            f"Decrease the length of {constraint.field_name} by at least {len(field_value) - constraint.max_characters} characters."
                        )
                    constraint_violations.append(" ".join(remedy_str))
                else:
                    self.print_verbose(
                        f"[PASS] Field {constraint.field_name} passed length validation: {len(field_value)} ({constraint.min_characters} - {constraint.max_characters})"
                    )

        return constraint_violations

    def wrap_task(self, task: str, result: str, violations: list[str]):
        wrapped_task = [
            "Your task was the following: \n\n" + task + "\n",
            "Your output was the following: \n\n" + result + "\n",
            "However, the output did violate the following constraints:",
        ]
        for constraint_violation in violations:
            wrapped_task.append(constraint_violation)
        wrapped_task.append(
            "\nFollow the original task and make sure to adhere to the constraints."
        )
        return "\n".join(wrapped_task)

    @staticmethod
    def get(obj, path: str):
        value = obj
        for i, key in enumerate(path.split(".")):
            if isinstance(value, list):
                try:
                    index = int(key)
                    if index < len(value):
                        value = value[index]
                    else:
                        raise ValueError(f"Index {index} out of range in {value}")
                except:
                    values = []
                    for val in value:
                        leaf = LengthConstrainedFunction.get(
                            val, ".".join(path.split(".")[i:])
                        )
                        values.extend(leaf)
                    return values
            elif isinstance(value, dict):
                if key in value:
                    value = value[key]
                else:
                    raise ValueError(f"Key {key} not found in {value}")
            else:
                if hasattr(value, key):
                    value = getattr(value, key)
                else:
                    raise ValueError(f"Key {key} not found in {value}")

        if not isinstance(value, list):
            value = [value]
        return value


class BaseStrategy(ValidatedFunction):
    def __init__(self, data_model: BaseModel, *args, **kwargs):
        super().__init__(
            data_model=data_model,
            retry_count=NUM_REMEDY_RETRIES,
            **kwargs,
        )
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        # TODO: inherit the strategy
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def forward(self, *args, **kwargs):
        result, _ = super().forward(
            *args,
            payload=self.payload,
            template_suffix=self.template,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return result

    @property
    def payload(self):
        return None

    @property
    def static_context(self):
        raise NotImplementedError()

    @property
    def template(self):
        return "{{fill}}"


class Strategy(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def __new__(self, module: str, *args, **kwargs):
        self._module = module
        self.module_path = f'symai.extended.strategies'
        return Strategy.load_module_class(self.module_path, self._module)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load_module_class(module_path, class_name):
        module_ = locate(module_path)
        return getattr(module_, class_name)
