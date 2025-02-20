import inspect
import logging
from pydoc import locate
from typing import Callable

import numpy as np
from beartype import beartype
from loguru import logger
from pydantic import BaseModel, ValidationError

from .components import Function
from .core_ext import deprecated
from .models import (
    ExceptionWithUsage,
    LengthConstraint,
    LLMDataModel,
    SemanticValidationError,
    TypeValidationError,
)
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
    Child classes must define their own forward() and remedy_prompt() methods.
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
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(retry_params=retry_params, verbose=verbose, *args, **kwargs)
        self.data_model = None

    def register_data_model(self, data_model: LLMDataModel, override=False):
        if self.data_model is not None and not override:
            raise ValueError("Data model already registered. If you want to override it, set `override=True`.")
        self.data_model = data_model

    def remedy_prompt(self, json: str, errors: str):
        """
        Override of base remedy_prompt giving specific context and instructions.
        """
        return (
            "The original JSON input was:\n"
            "<original_input>\n"
            "```json\n"
            f"{json}\n"
            "```\n"
            "</original_input>\n"
            "\n\n"
            "During type validation, the following errors were found:\n"
            "<validation_errors>\n"
            f"{errors}\n"
            "</validation_errors>\n"
            "\n\n"
            "The JSON must conform to this schema:\n"
            "<json_schema>\n"
            f"{self.data_model.instruct_llm()}\n"
            "</json_schema>\n"
        )

    def forward(self, *args, **kwargs):
        if self.data_model is None:
            raise ValueError("Data model is not registered. Please register the data model before calling the `forward` method.")

        # Force JSON mode
        kwargs["response_format"] = {"type": "json_object"}
        logger.info(f"Initializing type validation with JSON mode for the data model {self.data_model.simplify_json_schema()}…")

        # Initial guess
        json_str = super().forward(
            f"Return a valid type according to the following JSON schema:\n{self.data_model.simplify_json_schema()}",
            *args,
            **kwargs
        ).value

        remedy_seeds = self.prepare_seeds(self.retry_params["tries"], **kwargs)
        logger.info(f"Prepared {len(remedy_seeds)} remedy seeds for type validation attempts…")

        result = None
        last_error = ""

        for i in range(self.retry_params["tries"]):
            try:
                logger.info(f"Attempt {i+1}/{self.retry_params['tries']}: Attempting type validation…")
                # Try to validate against provided data model
                result = self.data_model.model_validate_json(json_str, strict=True)
                logger.info("Type validation successful!")
                break
            except ValidationError as e:
                logger.info(f"Type validation attempt {i+1} failed, pausing before retry…")
                self._pause()

                error_str = self.simplify_validation_errors(e)
                logger.info(f"Type validation errors identified: {error_str}!")

                # Change the dynamic context of the remedy function to account for the errors
                logger.info("Updating remedy function context…")
                context = self.remedy_prompt(json=json_str, errors=error_str)
                self.remedy_function.clear()
                self.remedy_function.adapt(context)
                json_str = self.remedy_function(seed=remedy_seeds[i]).value
                logger.info("Applied remedy function with updated context!")
                logger.info(f"New context: {self.remedy_function.dynamic_context}")

                last_error = error_str

        if result is None:
            logger.info(f"All type validation attempts failed. Last error: {last_error}!")
            raise TypeValidationError(f"Failed to retrieve valid JSON: {last_error}!")

        logger.info("Type validation completed successfully!")
        return result


class SemanticValidationFunction(ValidationFunction):
    """
    SemanticValidationFunction ensures that the final JSON (which conforms to the data schema)
    also passes additional semantic checks defined by user-provided conditions.
    """

    def __init__(
        self,
        retry_params: dict[str, int | float | bool] = ValidationFunction._default_retry_params,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(retry_params=retry_params, verbose=verbose, *args, **kwargs)
        self.input_data_model = None
        self.output_data_model = None

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
        """
        Override of base remedy_prompt providing instructions for fixing semantic validation errors.
        """
        return (
            "You are an expert in semantic validation. Your goal is to validate the output data model based on the prompt, "
            "the errors, and the output that produced the errors, and if given, the input data model and the input.\n"
            "Your prompt was:\n"
            "<prompt>\n"
            f"{prompt}\n"
            "</prompt>"
            "\n\n"
            "The input data model is:\n"
            "<input_data_model>\n"
            f"{self.input_data_model.simplify_json_schema() if self.input_data_model is not None else 'N/A'}\n"
            "</input_data_model>"
            "\n\n"
            "The given input was:\n"
            "<input>\n"
            f"{str(self.input_data_model) if self.input_data_model is not None else 'N/A'}\n"
            "</input>"
            "\n\n"
            "The output data model is:\n"
            "<output_data_model>\n"
            f"{self.output_data_model.instruct_llm()}\n"
            "</output_data_model>"
            "\n\n"
            "You've lastly generated the following output:\n"
            "<output>\n"
            f"{output}\n"
            "</output>"
            "\n\n"
            "During the semantic validation, the output was found to have the following errors:\n"
            "<errors>\n"
            f"{errors}\n"
            "</errors>"
            "\n\n"
            "You need to:\n"
            "1. Correct the provided output to address **all listed validation errors**.\n"
            "2. Ensure the corrected output adheres strictly to the requirements of the **original prompt**.\n"
            "3. Preserve the intended meaning and structure of the original prompt wherever possible.\n"
            "\n\n"
            "Important guidelines:\n"
            "</guidelines>\n"
            "- The result of the task must be the output data model.\n"
            "- Focus only on fixing the listed validation errors without introducing new errors or unnecessary changes.\n"
            "- Ensure the revised output is clear, accurate, and fully compliant with the original prompt.\n"
            "- Maintain proper formatting and any required conventions specified in the original prompt."
            "</guidelines>"
        )

    def zero_shot_prompt(self, prompt: str) -> str:
        return (
            "You are given the following prompt:\n"
            "<prompt>\n"
            f"{prompt}\n"
            "</prompt>"
            "\n\n"
            "The input data model is:\n"
            "<input_data_model>\n"
            f"{self.input_data_model.simplify_json_schema() if self.input_data_model is not None else 'N/A'}\n"
            "</input_data_model>"
            "\n\n"
            "The given input is:\n"
            "<input>\n"
            f"{str(self.input_data_model) if self.input_data_model is not None else 'N/A'}\n"
            "</input>"
            "\n\n"
            "The output data model is:\n"
            "<output_data_model>\n"
            f"{self.output_data_model.instruct_llm()}\n"
            "</output_data_model>"
            "\n\n"
            "Important guidelines:\n"
            "</guidelines>\n"
            "- The result of the task must be the output data model.\n"
            "- Focus only on fixing the listed validation errors without introducing new errors or unnecessary changes.\n"
            "- Ensure the revised output is clear, accurate, and fully compliant with the original prompt.\n"
            "- Maintain proper formatting and any required conventions specified in the original prompt."
            "</guidelines>"
        )

    def forward(self, prompt: str, f_semantic_conditions: list[Callable], *args, **kwargs):
        if self.output_data_model is None:
            raise ValueError("While the input data model is optional, the output data model must be provided. Please register it before calling the `forward` method.")

        # Force JSON mode
        kwargs["response_format"] = {"type": "json_object"}
        logger.info(
            f"Initializing semantic validation for prompt '{prompt}' with type validated input "
            f"{self.input_data_model.simplify_json_schema() if self.input_data_model else 'N/A'} "
            f"and type validated output {self.output_data_model.simplify_json_schema()}…"
        )

        # Zero shot the task
        context = self.zero_shot_prompt(prompt=prompt)
        json_str = super().forward(context, *args, **kwargs).value

        remedy_seeds = self.prepare_seeds(self.retry_params["tries"], **kwargs)
        logger.info(f"Prepared {len(remedy_seeds)} remedy seeds for semantic validation attempts…")

        result = None
        last_error = None
        errors = []

        for i in range(self.retry_params["tries"]):
            logger.info(f"Attempt {i+1}/{self.retry_params['tries']}: Attempting semantic validation…")
            try:
                result = self.output_data_model.model_validate_json(json_str, strict=True)
                if not all(f(result) for f in f_semantic_conditions):
                    raise ValueError("Semantic validation failed!")
                break
            except Exception as e:
                logger.error(f"Attempt {i+1} failed with error: {e}")
                self._pause()

                if isinstance(e, ValidationError):
                    error_str = self.simplify_validation_errors(e)
                    logger.info(f"The following errors were identified: {error_str}")
                else:
                    error_str = str(e)
                    logger.info(f"The following error was identified: {error_str}")
                errors.append(error_str)

                # Update remedy function context
                logger.info("Updating remedy function context…")
                context = self.remedy_prompt(prompt=prompt, output=json_str, errors="\n".join(errors))
                self.remedy_function.clear()
                self.remedy_function.adapt(context)
                json_str = self.remedy_function(seed=remedy_seeds[i], **kwargs).value
                logger.info("Applied remedy function with updated context!")
                logger.info(f"New context: {self.remedy_function.dynamic_context}")

                last_error = error_str

        if result is None or not all(f(result) for f in f_semantic_conditions):
            logger.info(f"Semantic validation attempts failed. Last error: {last_error}")
            raise SemanticValidationError(
                prompt=prompt,
                result=json_str,
                violations=errors,
            )

        logger.info("Semantic validation completed successfully!")
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
        remedy_retry_params: dict[str, int | float | bool] = _default_remedy_retry_params,
        verbose: bool = False
    ):
        '''
        A contract class decorator inspired by DbC principles, ensuring that the function's input and output
        adhere to specified data models. This implementation includes retry logic to handle transient errors
        and gracefully handle failures.

        Example usage in docstring...
        '''
        self.pre_remedy = pre_remedy
        self.post_remedy = post_remedy
        self.f_type_validation_remedy = TypeValidationFunction(verbose=False, retry_params=remedy_retry_params)
        self.f_semantic_validation_remedy = SemanticValidationFunction(verbose=False, retry_params=remedy_retry_params)

        if not verbose:
            logger.disable(__name__)
        else:
            logger.enable(__name__)

    def _is_valid_input(self, *args, **kwargs):
        logger.info("Validating input arguments...")
        if args:
            logger.error("Positional arguments detected!")
            raise ValueError("Positional arguments are not allowed. Please use keyword arguments instead.")
        input = kwargs.pop("input")
        if input is None:
            logger.error("No input argument provided!")
            raise ValueError("Please provide an `input` argument.")
        if not isinstance(input, LLMDataModel):
            logger.error(f"Invalid input type: {type(input)}")
            raise TypeError(f"Expected input to be of type `LLMDataModel`, got {type(input)}")
        logger.info("Input validation successful")
        return input

    def _validate_input(self, wrapped_self, input, **remedy_kwargs):
        if self.pre_remedy:
            logger.info("Validating pre-conditions with remedy...")
            if not hasattr(wrapped_self, 'pre'):
                logger.error("Pre-condition function not defined!")
                raise Exception("Pre-condition function not defined. Please define a `pre` method if you want to enforce pre-conditions through a remedy.")
            try:
                logger.info("Attempting pre-condition validation...")
                return wrapped_self.pre(input)
            except Exception as e:
                logger.error(f"Pre-condition validation failed: {str(e)}")
                logger.info("Attempting remedy with semantic validation...")
                self.f_semantic_validation_remedy.register_expected_data_model(input, attach_to="output")
                input = self.f_semantic_validation_remedy(wrapped_self.prompt, f_semantic_conditions=[wrapped_self.pre], **remedy_kwargs)
                logger.info("Semantic validation remedy successful")
                return input
        else:
            if hasattr(wrapped_self, 'pre'):
                logger.info("Validating pre-conditions without remedy...")
                if not wrapped_self.pre(input):
                    logger.error("Pre-conditions failed!")
                    raise Exception("Pre-conditions failed!")
                logger.info("Pre-conditions passed")
                return

    def _validate_output(self, wrapped_self, input, output, **remedy_kwargs):
        logger.info("Starting output validation...")
        try:
            logger.info("Registering output data model for type validation...")
            self.f_type_validation_remedy.register_data_model(output)
            output = self.f_type_validation_remedy()
            logger.info("Type validation successful")
        except Exception as e:
            logger.error(f"Type validation failed: {str(e)}")
            raise Exception("Type validation failed! Couldn't create a data model matching the output data model.")

        if self.post_remedy:
            logger.info("Validating post-conditions with remedy...")
            if not hasattr(wrapped_self, "post"):
                logger.error("Post-conditions not defined!")
                raise Exception("Post-conditions not defined. Please define a `post` attribute if you want to enforce semantic validation through a remedy.")

            logger.info("Setting up semantic validation...")
            self.f_semantic_validation_remedy.register_expected_data_model(input, attach_to="input")
            self.f_semantic_validation_remedy.register_expected_data_model(output, attach_to="output")
            output = self.f_semantic_validation_remedy(wrapped_self.prompt, f_semantic_conditions=[wrapped_self.post], **remedy_kwargs)
            logger.info("Semantic validation successful")
            return output
        else:
            if hasattr(wrapped_self, "post"):
                logger.info("Validating post-conditions without remedy...")
                if not wrapped_self.post(output):
                    logger.error("Semantic validation failed!")
                    raise Exception("Semantic validation failed!")
                logger.info("Post-conditions passed")
                return

    def __call__(self, cls):
        contract_self = self
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
            logger.info("Contract initialization complete")

        def wrapped_forward(wrapped_self, *args, **kwargs):
            logger.info("Starting contract forward pass...")
            original_input = contract_self._is_valid_input(*args, **kwargs)

            remedy_kwargs = dict(
                payload=getattr(wrapped_self, "payload", None),
                template_suffix=getattr(wrapped_self, "template", None)
            )

            sig = inspect.signature(original_forward)
            original_output_type = sig.return_annotation
            if original_output_type == inspect._empty:
                logger.error("Missing return type annotation!")
                raise ValueError("The contract requires a return type annotation.")
            if not issubclass(original_output_type, LLMDataModel):
                logger.error(f"Invalid return type: {original_output_type}")
                raise ValueError("The return type annotation must be a subclass of `LLMDataModel`.")

            try:
                input = original_input
                maybe_new_input = contract_self._validate_input(wrapped_self, input, **remedy_kwargs)
                if maybe_new_input is not None:
                    input = maybe_new_input

                output = self._validate_output(wrapped_self, input, original_output_type, **remedy_kwargs)
                wrapped_self.contract_successful = True
                wrapped_self.contract_result = output
            finally:
                # Execute the original forward method
                logger.info("Executing original forward method...")
                kwargs['input'] = original_input
                output = original_forward(wrapped_self, *args, **kwargs)

                if not isinstance(output, original_output_type):
                    logger.error(f"Output type mismatch: {type(output)}")
                    raise TypeError(
                        f"Expected output to be an instance of {original_output_type}, "
                        f"but got {type(output)}! Forward method must return an instance of {original_output_type}!"
                    )
                if not wrapped_self.contract_successful:
                    logger.warning("Contract validation failed, checking output type...")
                else:
                    logger.success("Contract validation successful")

            return output

        cls.__init__ = __init__
        cls.forward = wrapped_forward
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
