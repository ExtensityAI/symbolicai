import json
import inspect
import logging
import time
from collections import defaultdict
from pydoc import locate
from typing import Callable

import numpy as np
from beartype import beartype
from loguru import logger
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape

from .components import Function
from .core_ext import deprecated
from .models import (ExceptionWithUsage, LengthConstraint, LLMDataModel,
                     SemanticValidationError, TypeValidationError)
from .symbol import Expression

NUM_REMEDY_RETRIES = 10


class ValidationFunction(Function):
    """
    Base class for validation functions that share common logic for:
      • Retry attempts
      • Remedy function
      • Prepare seeds
      • Pause/backoff logic
      • Error simplification
    """

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
        retry_params: dict[str, int | float | bool] = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if retry_params is None:
            retry_params = self._default_retry_params
        self.retry_params = retry_params
        self.console = Console()

        # Standard remedy function for JSON correction:
        self.remedy_function = Function(
            "Fix the provided JSON string to ensure it is valid according to the schema and resolves all listed validation errors.",
            static_context="""
            <goal>
            You are tasked with fixing a string that is intended to be in **JSON format** but contains errors.
            Your goal is to correct the errors and ensure the JSON string is valid according to a given JSON schema.
            </goal>

            It is is very important to follow these guidelines and requirements:
            <guidelines>
            1. Parse the provided string and use the list of validation errors to identify what needs to be fixed.
            2. Correct the identified errors to produce a properly formatted JSON string.
            3. Ensure the corrected JSON complies fully with the provided JSON schema.
            4. Preserve all original keys and values as much as possible. Only modify keys or values if they do not comply with the schema.
            5. Only modify the structure or values if necessary to meet the schema's requirements.
            6. Return the corrected JSON string as the output.
            </guidelines>

            <requirements>
            - The output must be a valid, well-formatted JSON string.
            - Do not introduce new data or alter the intent of the original content unless required for schema compliance.
            - Ensure all changes are minimal and strictly necessary to fix the listed errors.
            </requirements>
            """,
            response_format={"type": "json_object"},
        )

        if not verbose:
            logger.disable(__name__)
        else:
            logger.enable(__name__)

    def prepare_seeds(self, num_seeds: int, **kwargs):
        # Get list of seeds for remedy (to avoid same remedy for same input)
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
            field_path = " -> ".join([str(element) for element in err["loc"]])
            message = err["msg"]
            expected_type = err.get("type", "unknown")
            provided_value = err.get("ctx", {}).get("given", "unknown")

            error_message = (
                f"Field '{field_path}': {message}. "
                f"Expected type: {expected_type}. Provided value: {provided_value}."
            )
            simplified_errors.append(error_message)

        return "\n".join(simplified_errors)

    def _pause(self):
        """
        Pause logic with backoff and jitter.
        """
        _delay = self.retry_params['delay']
        _delay *= self.retry_params['backoff']

        if isinstance(self.retry_params['jitter'], tuple):
            _delay += np.random.uniform(*self.retry_params['jitter'])
        else:
            _delay += self.retry_params['jitter']

        if self.retry_params['max_delay'] >= 0:
            _delay = min(_delay, self.retry_params['max_delay'])

    def remedy_prompt(self, *args, **kwargs):
        """
        Abstract or base remedy prompt method.
        Child classes typically override this to include additional context needed for correction.
        """
        raise NotImplementedError("Each child class needs its own remedy_prompt implementation.")

    def display_panel(self, content, title, border_style="cyan", style="#f0eee6", padding=(1,2)):
        """
        Display content in a rich panel with consistent formatting.

        Args:
            content: The content to display in the panel
            title: The title of the panel
            border_style: Color of the panel border (default: "cyan")
            style: Style of the panel content (default: "#f0eee6")
            padding: Padding for the panel (default: (1,2))
        """
        body = escape(content)
        panel = Panel.fit(body, title=title, padding=padding, border_style=border_style, style=style)
        self.console.print(panel)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs) # Just propagate to Function


class TypeValidationFunction(ValidationFunction):
    """
    TypeValidationFunction ensures the output is valid JSON matching a specific data model (LLMDataModel).
    If validation fails, it attempts to fix it up to a certain number of retries.
    """

    def __init__(
        self,
        retry_params: dict[str, int | float | bool] = ValidationFunction._default_retry_params,
        accumulate_errors: bool = False,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(retry_params=retry_params, verbose=verbose, *args, **kwargs)
        self.data_model = None
        self.accumulate_errors = accumulate_errors
        self.verbose = verbose

    def register_data_model(self, data_model: LLMDataModel, override=False):
        if self.data_model is not None and not override:
            raise ValueError("Data model already registered. If you want to override it, set `override=True`.")
        self.data_model = data_model

    def remedy_prompt(self, json: str, errors: str):
        """
        Override of base remedy_prompt giving specific context and instructions.
        """
        return f"""
The original JSON input was:
<original_input>
```json
{json}
```
</original_input>

During type validation, the following errors were found:
<validation_errors>
{errors}
</validation_errors>

The JSON must conform to this schema:
<json_schema>
{self.data_model.instruct_llm()}
</json_schema>
"""

    def forward(self, *args, **kwargs):
        if self.data_model is None:
            raise ValueError("Data model is not registered. Please register the data model before calling the `forward` method.")
        # Force JSON mode
        kwargs["response_format"] = {"type": "json_object"}
        logger.info(f"Initializing type validation…")
        if self.verbose:
            self.display_panel(
                self.data_model.simplify_json_schema(),
                title="Data Model"
            )

        # Initial guess
        json_str = super().forward(
            f"Return a valid type according to the following JSON schema:\n{self.data_model.simplify_json_schema()}",
            *args,
            **kwargs
        ).value

        remedy_seeds = self.prepare_seeds(self.retry_params["tries"], **kwargs)
        logger.info(f"Prepared {len(remedy_seeds)} remedy seeds for type validation attempts…")

        result = None
        errors = []

        for i in range(self.retry_params["tries"]):
            try:
                logger.info(f"Attempt {i+1}/{self.retry_params['tries']}: Attempting type validation…")
                # Try to validate against provided data model
                result = self.data_model.model_validate_json(json_str, strict=True)
                break
            except ValidationError as e:
                logger.info(f"Type validation attempt {i+1} failed, pausing before retry…")

                self._pause()

                error_str = self.simplify_validation_errors(e)
                logger.error(f"Type validation errors identified!")
                if self.verbose:
                    self.display_panel(
                        "\n".join(errors) if self.accumulate_errors else error_str,
                        title=f"Type Validation Errors ({'accumulated errors' if self.accumulate_errors else 'last error'})",
                        border_style="red"
                    )

                errors.append(error_str)

                # Change the dynamic context of the remedy function to account for the errors
                logger.info("Updating remedy function context…")
                context = self.remedy_prompt(json=json_str, errors="\n".join(errors) if self.accumulate_errors else error_str)
                self.remedy_function.clear()
                self.remedy_function.adapt(context)
                json_str = self.remedy_function(seed=remedy_seeds[i], **kwargs).value
                logger.info("Applied remedy function with updated context!")
                if self.verbose:
                    self.display_panel(
                        self.remedy_function.dynamic_context,
                        title="New Context"
                    )

        if result is None:
            logger.error(f"All type validation attempts failed!")
            raise TypeValidationError(f"Failed to retrieve valid JSON!")

        logger.success("Type validation completed successfully!")
        # Clear artifacts from the remedy function
        self.remedy_function.clear()

        return result


class SemanticValidationFunction(ValidationFunction):
    """
    SemanticValidationFunction ensures that the final JSON (which conforms to the data schema)
    also passes additional semantic checks defined by user-provided conditions.
    """

    def __init__(
        self,
        retry_params: dict[str, int | float | bool] = ValidationFunction._default_retry_params,
        accumulate_errors: bool = False,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(retry_params=retry_params, verbose=verbose, *args, **kwargs)
        self.input_data_model = None
        self.output_data_model = None
        self.accumulate_errors = accumulate_errors
        self.verbose = verbose

    def register_expected_data_model(self, data_model: LLMDataModel, attach_to: str, override: bool = False):
        assert attach_to in ["input", "output"], f"Invalid attach_to value: {attach_to}; must be either 'input' or 'output'"
        if attach_to == "input":
            if self.input_data_model is not None and not override:
                raise ValueError("There is already a data model attached to the input. If you want to override it, set `override=True`.")
            self.input_data_model = data_model
        elif attach_to == "output":
            if self.output_data_model is not None and not override:
                raise ValueError("There is already a data model attached to the output. If you want to override it, set `override=True`.")
            self.output_data_model = data_model

    def remedy_prompt(self, prompt: str, output: str, errors: str):
        """Override of base remedy_prompt providing instructions for fixing semantic validation errors."""
        return f"""
You are an expert in semantic validation. Your goal is to validate the output data model based on the prompt, the errors, and the output that produced the errors, and if given, the input data model and the input.

Your prompt was:
<prompt>
{prompt}
</prompt>

The input data model is:
<input_data_model>
{self.input_data_model.simplify_json_schema() if self.input_data_model is not None else 'N/A'}
</input_data_model>

The given input was:
<input>
{str(self.input_data_model) if self.input_data_model is not None else 'N/A'}
</input>

The output data model is:
<output_data_model>
{self.output_data_model.instruct_llm()}
</output_data_model>

You've lastly generated the following output:
<output>
{output}
</output>

During the semantic validation, the output was found to have the following errors:
<errors>
{errors}
</errors>

You need to:
1. Correct the provided output to address **all listed validation errors**.
2. Ensure the corrected output adheres strictly to the requirements of the **original prompt**.
3. Preserve the intended meaning and structure of the original prompt wherever possible.

Important guidelines:
</guidelines>
- The result of the task must be the output data model.
- Focus only on fixing the listed validation errors without introducing new errors or unnecessary changes.
- Ensure the revised output is clear, accurate, and fully compliant with the original prompt.
- Maintain proper formatting and any required conventions specified in the original prompt.
</guidelines>
"""

    def zero_shot_prompt(self, prompt: str) -> str:
        """We try to zero-shot the task, maybe we're lucky!"""
        return f"""
You are given the following prompt:
<prompt>
{prompt}
</prompt>

The input data model is:
<input_data_model>
{self.input_data_model.simplify_json_schema() if self.input_data_model is not None else 'N/A'}
</input_data_model>

The given input is:
<input>
{str(self.input_data_model) if self.input_data_model is not None else 'N/A'}
</input>

The output data model is:
<output_data_model>
{self.output_data_model.instruct_llm()}
</output_data_model>

Important guidelines:
</guidelines>
- The result of the task must be the output data model.
- Focus only on fixing the listed validation errors without introducing new errors or unnecessary changes.
- Ensure the revised output is clear, accurate, and fully compliant with the original prompt.
- Maintain proper formatting and any required conventions specified in the original prompt.
</guidelines>
"""

    def forward(self, prompt: str, f_semantic_conditions: list[Callable], *args, **kwargs):
        if self.output_data_model is None:
            raise ValueError("While the input data model is optional, the output data model must be provided. Please register it before calling the `forward` method.")
        validation_context = kwargs.pop('validation_context', {})
        # Force JSON mode
        kwargs["response_format"] = {"type": "json_object"}
        logger.info("Initializing semantic validation…")
        if self.verbose:
            for label, body in [
                ("Prompt", prompt),
                ("Input data model", self.input_data_model.simplify_json_schema() if self.input_data_model else 'N/A'),
                ("Output data model", self.output_data_model.simplify_json_schema()),
            ]:
                self.display_panel(body, title=label)

        # Zero shot the task
        context = self.zero_shot_prompt(prompt=prompt)
        json_str = super().forward(context, *args, **kwargs).value

        remedy_seeds = self.prepare_seeds(self.retry_params["tries"], **kwargs)
        logger.info(f"Prepared {len(remedy_seeds)} remedy seeds for semantic validation attempts…")

        result = None
        errors = []
        for i in range(self.retry_params["tries"]):
            logger.info(f"Attempt {i+1}/{self.retry_params['tries']}: Attempting semantic validation…")
            try:
                result = self.output_data_model.model_validate_json(json_str, strict=True, context = validation_context)
                if not all(f(result) for f in f_semantic_conditions):
                    raise ValueError("Semantic validation failed!")
                break
            except Exception as e:
                logger.info(f"Semantic validation attempt {i+1} failed, pausing before retry…")

                self._pause()

                if isinstance(e, ValidationError):
                    error_str = self.simplify_validation_errors(e)
                else:
                    error_str = str(e)

                errors.append(error_str)

                logger.error(f"Semantic validation errors identified!")
                if self.verbose:
                    self.display_panel(
                        "\n".join(errors) if self.accumulate_errors else error_str,
                        title=f"Semantic Validation Errors ({'accumulated errors' if self.accumulate_errors else 'last error'})",
                        border_style="red"
                    )

                # Update remedy function context
                logger.info("Updating remedy function context…")
                context = self.remedy_prompt(prompt=prompt, output=json_str, errors="\n".join(errors) if self.accumulate_errors else error_str)
                self.remedy_function.clear()
                self.remedy_function.adapt(context)
                if self.verbose:
                    self.display_panel(
                        self.remedy_function.dynamic_context,
                        title="New Context"
                    )

                # Apply the remedy function
                json_str = self.remedy_function(seed=remedy_seeds[i], **kwargs).value
                logger.info("Applied remedy function with updated context!")

        if result is None or not all(f(result) for f in f_semantic_conditions):
            logger.error(f"All semantic validation attempts failed!")
            raise SemanticValidationError(
                prompt=prompt,
                result=json_str,
                violations=errors,
            )

        logger.success("Semantic validation completed successfully!")
        # Clear artifacts from the remedy function
        self.remedy_function.clear()

        return result


@beartype
class contract:
    _default_remedy_retry_params = dict(
        tries=5,
        delay=0.5,
        max_delay=15,
        jitter=0.1,
        backoff=2,
        graceful=False
    )

    def __init__(
        self,
        pre_remedy: bool = False,
        post_remedy: bool = True,
        accumulate_errors: bool = False,
        verbose: bool = False,
        remedy_retry_params: dict[str, int | float | bool] = _default_remedy_retry_params,
    ):
        '''
        A contract class decorator inspired by DbC principles. It ensures that the function's input and output
        adhere to specified data models both syntactically and semantically. This implementation includes retry
        logic to handle transient errors and gracefully handle failures.
        '''
        self.pre_remedy = pre_remedy
        self.post_remedy = post_remedy
        self.f_type_validation_remedy = TypeValidationFunction(accumulate_errors=accumulate_errors, verbose=verbose, retry_params=remedy_retry_params)
        self.f_semantic_validation_remedy = SemanticValidationFunction(accumulate_errors=accumulate_errors, verbose=verbose, retry_params=remedy_retry_params)

        if not verbose:
            logger.disable(__name__)
        else:
            logger.enable(__name__)

    def _is_valid_input(self, *args, **kwargs):
        logger.info("Validating input argument...")
        if args:
            logger.error("Positional arguments detected!")
            raise ValueError("Positional arguments are not allowed! Please use keyword arguments instead.")
        input = kwargs.pop("input")
        if input is None:
            logger.error("No input argument provided!")
            raise ValueError("Please provide an `input` argument.")
        if not isinstance(input, LLMDataModel):
            logger.error(f"Invalid input type: {type(input)}")
            raise TypeError(f"Expected input to be of type `LLMDataModel`, got {type(input)}")
        logger.success("Input argument validated successfully!")
        return input

    def _validate_input(self, wrapped_self, input, it, **remedy_kwargs):
        logger.info("Starting input validation...")
        if self.pre_remedy:
            logger.info("Validating pre-conditions with remedy...")
            if not hasattr(wrapped_self, 'pre'):
                logger.error("Pre-condition function not defined!")
                raise Exception("Pre-condition function not defined. Please define a `pre` method if you want to enforce pre-conditions through a remedy.")

            try:
                assert wrapped_self.pre(input)
                logger.success("Pre-condition validation successful!")
                return input
            except Exception as e:
                logger.error("Pre-condition validation failed!")
                try:
                    op_start = time.perf_counter()
                    self.f_semantic_validation_remedy.register_expected_data_model(input, attach_to="output", override=True)
                    input = self.f_semantic_validation_remedy(wrapped_self.prompt, f_semantic_conditions=[wrapped_self.pre], **remedy_kwargs)
                finally:
                    wrapped_self._contract_timing[it]["input_semantic_validation"] = time.perf_counter() - op_start
                return input
        else:
            if hasattr(wrapped_self, 'pre'):
                logger.info("Validating pre-conditions without remedy...")
                op_start = time.perf_counter()
                try:
                    wrapped_self.pre(input) # Call pre; if it fails, it should raise an exception.
                except Exception as e:
                    # pre() itself raised an exception
                    logger.error(f"Pre-condition validation failed (exception in pre): {str(e)}")
                    # Raise the consistent failure exception, chaining the original for context
                    raise Exception("Pre-condition validation failed!") from e
                finally:
                    wrapped_self._contract_timing[it]["input_semantic_validation"] = time.perf_counter() - op_start

                logger.success("Pre-condition validation successful!")
                # On success for pre_remedy=False, return None, as input is not modified.
                return

    def _validate_output(self, wrapped_self, input, output, it, **remedy_kwargs):
        logger.info("Starting output validation...")
        try:
            try:
                op_start = time.perf_counter()
                self.f_type_validation_remedy.register_data_model(output, override=True)
                output = self.f_type_validation_remedy(**remedy_kwargs)
            finally:
                wrapped_self._contract_timing[it]["output_type_validation"] = time.perf_counter() - op_start
        except Exception as e:
            logger.error(f"Type validation failed: {str(e)}")
            raise Exception("Type validation failed! Couldn't create a data model matching the output data model.")

        if self.post_remedy:
            logger.info("Validating post-conditions with remedy...")
            if not hasattr(wrapped_self, "post"):
                logger.error("Post-condition function not defined!")
                raise Exception("Post-condition function not defined. Please define a `post` method if you want to enforce post-conditions through a remedy.")

            try:
                try:
                    assert wrapped_self.post(output)
                    logger.success("Post-condition validation successful!")
                    return output
                except Exception as e:
                    logger.error("Post-condition validation failed!")

                op_start = time.perf_counter()
                self.f_semantic_validation_remedy.register_expected_data_model(input, attach_to="input", override=True)
                self.f_semantic_validation_remedy.register_expected_data_model(output, attach_to="output", override=True)
                output = self.f_semantic_validation_remedy(wrapped_self.prompt, f_semantic_conditions=[wrapped_self.post], **remedy_kwargs)
            finally:
                wrapped_self._contract_timing[it]["output_semantic_validation"] = time.perf_counter() - op_start
            logger.success("Post-condition validation successful!")
            return output
        else:
            if hasattr(wrapped_self, "post"):
                logger.info("Validating post-conditions without remedy...")
                try:
                    op_start = time.perf_counter()
                    res = wrapped_self.post(output)
                finally:
                    wrapped_self._contract_timing[it]["output_semantic_validation"] = time.perf_counter() - op_start
                if not res:
                    logger.error("Post-condition validation failed!")
                    raise Exception("Post-condition validation failed!")
                logger.success("Post-condition validation successful!")
                return

    def _act(self, wrapped_self, input, it, **act_kwargs):
        act_method = getattr(wrapped_self, 'act', None)
        if callable(act_method):
            logger.info(f"Executing 'act' method on {wrapped_self.__class__.__name__}…")

            # Validate act_method signature
            try:
                act_sig = inspect.signature(act_method)
                act_params_list = list(act_sig.parameters.values())
            except ValueError:
                raise TypeError(f"Cannot inspect signature of 'act' method on {wrapped_self.__class__.__name__}.")

            # Validate input parameter type
            if len(act_params_list) <= 1:  # Need at least self and one input parameter
                raise TypeError(f"'act' method on {wrapped_self.__class__.__name__} must accept at least one input parameter after 'self'.")

            # Find the input parameter named 'input'
            input_param = None
            for param in act_params_list:
                if param.name == 'input' and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    input_param = param
                    break

            if input_param is None:
                raise TypeError(f"'act' method on {wrapped_self.__class__.__name__} must have a parameter named 'input'.")

            act_input_annotation = input_param.annotation
            if act_input_annotation == inspect.Parameter.empty:
                raise TypeError(f"The input parameter '{input_param.name}' of 'act' method on {wrapped_self.__class__.__name__} must have a type annotation.")

            if not (inspect.isclass(act_input_annotation) and issubclass(act_input_annotation, LLMDataModel)):
                raise TypeError(f"The input type annotation for 'act' method on {wrapped_self.__class__.__name__} must be a subclass of LLMDataModel.")

            # Validate return type
            act_return_annotation = act_sig.return_annotation
            if act_return_annotation == inspect.Signature.empty:
                raise TypeError(f"'act' method on {wrapped_self.__class__.__name__} must have a return type annotation.")

            if not (inspect.isclass(act_return_annotation) and issubclass(act_return_annotation, LLMDataModel)):
                raise TypeError(f"The return type annotation '{act_return_annotation}' for 'act' method on {wrapped_self.__class__.__name__} must be a subclass of LLMDataModel.")

            try:
                op_start_act = time.perf_counter()
                act_output = act_method(input, **act_kwargs)
            finally:
                wrapped_self._contract_timing[it]["act_execution"] = time.perf_counter() - op_start_act

            # Validate act output type
            if not isinstance(act_output, act_return_annotation):
                raise TypeError(f"'act' method returned {type(act_output).__name__}, expected {act_return_annotation.__name__}.")

            logger.success("'act' method executed successfully!")
            return act_output
        # Propagate the input if no act method is defined
        return input

    def __call__(self, cls):
        original_init = cls.__init__
        original_forward = cls.forward

        def __init__(wrapped_self, *args, **kwargs):
            logger.info("Initializing contract...")
            original_init(wrapped_self, *args, **kwargs)

            if not hasattr(wrapped_self, "prompt"):
                logger.error("Prompt attribute not defined!")
                raise Exception("Prompt not defined. Please define a `prompt` attribute to enforce semantic validation.")

            wrapped_self.contract_successful = False
            wrapped_self.contract_result = None
            wrapped_self._contract_timing = defaultdict(dict)
            logger.info("Contract initialization complete!")

        def wrapped_forward(wrapped_self, *args, **kwargs):
            it = len(wrapped_self._contract_timing) # the len is the __call__ op_start
            contract_start = time.perf_counter()
            logger.info("Starting contract execution...")
            try:
                op_start = time.perf_counter()
                original_input = self._is_valid_input(*args, **kwargs)
            finally:
                wrapped_self._contract_timing[it]["input_type_validation"] = time.perf_counter() - op_start

            maybe_payload = getattr(wrapped_self, "payload", None)
            maybe_template = getattr(wrapped_self, "template")
            if inspect.ismethod(maybe_template):
                # `template` is a primitive in symbolicai case in which we actually don't have a template
                maybe_template = None

            # Create validation kwargs that include all original kwargs plus payload and template
            validation_kwargs = {
                **kwargs,
                "payload": maybe_payload,
                "template_suffix": maybe_template
            }
            validation_kwargs_filtered = {k: v for k, v in validation_kwargs.items() if k != "input"}
            act_kwargs = {k: v for k, v in kwargs.items() if k != 'input'}

            sig = inspect.signature(original_forward)
            original_output_type = sig.return_annotation
            if original_output_type == inspect._empty:
                logger.error("Missing return type annotation!")
                raise ValueError("The contract requires a return type annotation.")
            if not issubclass(original_output_type, LLMDataModel):
                logger.error(f"Invalid return type: {original_output_type}")
                raise ValueError("The return type annotation must be a subclass of `LLMDataModel`.")

            final_output = None
            try:
                # 1. Start with original input and apply pre-validation
                current_input = original_input
                maybe_new_input = self._validate_input(wrapped_self, current_input, it, **validation_kwargs_filtered)
                if maybe_new_input is not None:
                    current_input = maybe_new_input

                # 2. Check if 'act' method exists and execute it
                current_input = self._act(wrapped_self, current_input, it, **act_kwargs)

                # 3. Validate output type and prepare for original_forward
                output = self._validate_output(wrapped_self, current_input, original_output_type, it, **validation_kwargs_filtered)
                wrapped_self.contract_successful = True
                wrapped_self.contract_result = output
                final_output = output # Output from the successful contract path

            except Exception as e:
                logger.error(f"Contract execution failed in main path: {str(e)}")
                wrapped_self.contract_successful = False
                # contract_result remains None or its value before the exception.
                # final_output remains None or its value before the exception.
                # The finally block's execution of original_forward will determine the actual returned value.
            finally:
                # Execute the original forward method with appropriate input
                logger.info("Executing original forward method...")

                # If contract was successful, use the processed input (after pre-validation and act, both optional)
                # `current_input` at this stage is the result of the try block's processing up to the point of exception,
                # or the full processing if successful.
                # If contract failed, use original_input (fallback).
                forward_input = current_input if wrapped_self.contract_successful else original_input

                # Prepare kwargs for original_forward
                forward_kwargs = kwargs.copy()
                forward_kwargs['input'] = forward_input

                try:
                    op_start = time.perf_counter()
                    output = original_forward(wrapped_self, **forward_kwargs)
                finally:
                    wrapped_self._contract_timing[it]["forward_execution"] = time.perf_counter() - op_start
                wrapped_self._contract_timing[it]["contract_execution"] = time.perf_counter() - contract_start

                if not isinstance(output, original_output_type):
                    logger.error(f"Output type mismatch: {type(output)}")
                    raise TypeError(
                        f"Expected output to be an instance of {original_output_type}, "
                        f"but got {type(output)}! Forward method must return an instance of {original_output_type}!"
                    )
                if not wrapped_self.contract_successful:
                    logger.warning("Contract validation failed!")
                else:
                    logger.success("Contract validation successful!")

                # Use the output from original_forward
                final_output = output
            return final_output

        def contract_perf_stats(wrapped_self):
            """Analyzes and prints timing statistics across all forward calls."""
            console = Console()

            num_calls = len(wrapped_self._contract_timing)
            if num_calls == 0:
                console.print("No contract executions recorded.")
                return {}

            ordered_operations = [
                "input_type_validation",
                "input_semantic_validation",
                "act_execution",
                "output_type_validation",
                "output_semantic_validation",
                "forward_execution",
                "contract_execution"
            ]

            stats = {}
            for op in ordered_operations:
                times = []
                for timing in wrapped_self._contract_timing.values():
                    if op in timing:
                        times.append(timing[op])
                    else:
                        times.append(0.0)

                non_zero_times = [t for t in times if t > 0]
                actual_count = len(non_zero_times)

                total_time = sum(times)
                mean_time = np.mean(non_zero_times) if non_zero_times else 0
                std_time = np.std(non_zero_times) if len(non_zero_times) > 1 else 0
                min_time = min(non_zero_times) if non_zero_times else 0
                max_time = max(non_zero_times) if non_zero_times else 0

                stats[op] = {
                    'count': actual_count,
                    'total': total_time,
                    'mean': mean_time,
                    'std': std_time,
                    'min': min_time,
                    'max': max_time
                }

            total_execution_time = stats['contract_execution']['total']
            for op in ordered_operations[:-1]:
                if total_execution_time > 0:
                    stats[op]['percentage'] = (stats[op]['total'] / total_execution_time) * 100
                else:
                    stats[op]['percentage'] = 0
            stats['contract_execution']['percentage'] = 100.0

            table = Table(
                title=f"Contract Execution Summary ({num_calls} Forward Calls)",
                show_header=True
            )
            table.add_column("Operation", style="cyan")
            table.add_column("Count", justify="right", style="blue")
            table.add_column("Total Time (s)", justify="right", style="green")
            table.add_column("Mean (s)", justify="right", style="yellow")
            table.add_column("Std Dev (s)", justify="right", style="magenta")
            table.add_column("Min (s)", justify="right", style="red")
            table.add_column("Max (s)", justify="right", style="red")
            table.add_column("% of Total", justify="right", style="cyan")

            for op in ordered_operations[:-1]:
                s = stats[op]
                table.add_row(
                    op.replace("_", " ").title(),
                    str(s['count']),
                    f"{s['total']:.3f}",
                    f"{s['mean']:.3f}",
                    f"{s['std']:.3f}",
                    f"{s['min']:.3f}",
                    f"{s['max']:.3f}",
                    f"{s['percentage']:.1f}%"
                )

            s = stats['contract_execution']
            table.add_row(
                "Total Execution",
                "N/A",
                f"{s['total']:.3f}",
                f"{s['mean']:.3f}",
                f"{s['std']:.3f}",
                f"{s['min']:.3f}",
                f"{s['max']:.3f}",
                "100.0%",
                style="bold magenta"
            )

            console.print("\n")
            console.print(table)
            console.print("\n")

            return stats

        cls.__init__ = __init__
        cls.forward = wrapped_forward
        cls.contract_perf_stats = contract_perf_stats

        return cls


class BaseStrategy(TypeValidationFunction):
    def __init__(self, data_model: BaseModel, *args, **kwargs):
        super().__init__(
            retry_params=dict(tries=NUM_REMEDY_RETRIES, delay=0.5, max_delay=15, backoff=2, jitter=0.1, graceful=False),
            **kwargs,
        )
        super().register_data_model(data_model)
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def forward(self, *args, **kwargs):
        result = super().forward(
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
        self.module_path = 'symai.extended.strategies'
        return Strategy.load_module_class(self.module)
