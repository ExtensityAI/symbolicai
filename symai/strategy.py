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
from .models import (ExceptionWithUsage, LengthConstraint, LLMDataModel,
                     SemanticValidationError, TypeValidationError)
from .symbol import Expression

NUM_REMEDY_RETRIES = 10

class TypeValidationFunction(Function):
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
        retry_params: dict[str, int | float | bool] = _default_retry_params,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retry_params = retry_params
        self.data_model = None
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

    def register_data_model(self, data_model: LLMDataModel, override=False):
        if self.data_model is not None and not override:
            raise ValueError("Data model already registered. If you want to override it, set `override=True`.")
        self.data_model = data_model

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
        if self.data_model is None:
            raise ValueError("Data model is not registered. Please register the data model before calling the `forward` method.")

        # Force JSON mode
        kwargs["response_format"] = {"type": "json_object"}
        logger.info(f"Initializing type validation with JSON mode for the data model {self.data_model.simplify_json_schema()}…")
        # Initial guess
        json = super().forward(f"Return a valid type according to the following JSON schema:\n{self.data_model.simplify_json_schema()}", *args, **kwargs).value
        # Get list of seeds for remedy (to avoid same remedy for same input)
        remedy_seeds = self.prepare_seeds(self.retry_params["tries"], **kwargs)
        logger.info(f"Prepared {len(remedy_seeds)} remedy seeds for type validation attempts…")

        result = None
        last_error = ""
        for i in range(self.retry_params["tries"]):
            try:
                logger.info(f"Attempt {i+1}/{self.retry_params['tries']}: Attempting type validation…")
                # Try to validate against provided data model
                result = self.data_model.model_validate_json(json, strict=True)
                logger.info("Type validation successful!")
                break
            except ValidationError as e:
                logger.info(f"Type validation attempt {i+1} failed, pausing before retry…")
                self._pause()

                error_str = self.simplify_validation_errors(e)
                logger.info(f"Type validation errors identified: {error_str}!")

                # Change the dynamic context of the remedy function to account for the errors
                logger.info("Updating remedy function context…")
                self.remedy_function.clear()
                self.remedy_function.adapt(f"[Original Input]\n```json\n{json}\n´´´\n")
                self.remedy_function.adapt(f"[Validation Errors]\n{error_str}\n")
                self.remedy_function.adapt(f"[JSON Schema]{self.data_model.instruct_llm()}\n")
                json = self.remedy_function(seed=remedy_seeds[i]).value
                logger.info("Applied remedy function with updated context!")
                logger.info(f"New context: {self.remedy_function.dynamic_context}")

                last_error = error_str

        if result is None:
            logger.info(f"All type validation attempts failed. Last error: {last_error}!")
            raise TypeValidationError(f"Failed to retrieve valid JSON: {last_error}!")

        logger.info("Type validation completed successfully!")
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


class SemanticValidationFunction(Function):
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
        retry_params: dict[str, int | float | bool] = _default_retry_params,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        A contract class decorator inspired by DbC principles, ensuring that the function's input and output adhere to specified data models.
        This implementation includes retry logic to handle transient errors and gracefully handle failures.

        Example:
            class CodeSnippet(LLMDataModel):
                code: str = Field(description="The code snippet that needs to be reviewed. It will be reviewed for security vulnerabilities, performance issues, and code complexity.")
                context: str = Field(default="Security vulnerability audit.", description="The context of the code.")
                language: str = Field(default="python", description="The programming language.")

            class CodeReview(LLMDataModel):
                code: CodeSnippet
                issues: list[str] = Field(default=[], description="A list of issues found in the code.")
                suggestions: list[str] = Field(default=[], description="A list of suggestions to improve the code.")
                security_concerns: list[str] = Field(default=[], description="A list of security concerns found in the code.")
                complexity_score: int = Field(default=0, ge=1, le=10, description="The complexity score of the code.")

            class AdaptedCodeSnippet(LLMDataModel):
                code: str = Field(description="The code snippet that was changed to improve security, performance, and complexity.")
                passed: bool = Field(description="Whether the code passed the review.")

            @contract(pre_remedy=False, post_remedy=True, verbose=True)
            class CodeReviewer(Expression):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def forward(self, input: CodeReview, **kwargs) -> AdaptedCodeSnippet:
                    pass

                #——————————————————————————————————————————————————————————————————————————————
                # called in the contract; must be implemented if pre_remedy is True
                def pre(self, input: CodeReview) -> bool:
                    return True

                # called in the contract; must be implemented if post_remedy is True
                def post(self, output: AdaptedCodeSnippet) -> bool:
                    if output.passed:
                        return True
                    return False
                #——————————————————————————————————————————————————————————————————————————————
                @property
                def task(self) -> str:
                    return "You are a code review assistant and your task is to analyze the provided code. Use your knowledge of Python and its ecosystem to identify potential security vulnerabilities, performance issues, and code quality concerns. If there are any issues with the provided code, you have to rewrite it to address the identified concerns."


            code = '''
            import pickle

            serialized_data = input("Enter serialized data: ")

            deserialized_data = pickle.loads(serialized_data.encode('latin1'))  # Unsafe deserialization
            '''

            input = CodeReview(code=CodeSnippet(code=code))
            code_reviewer = CodeReviewer()
            res = code_reviewer(input=input)
        """
        super().__init__(*args, **kwargs)
        self.retry_params = retry_params
        self.input_data_model = None
        self.output_data_model = None
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

    def register_input_data_model(self, data_model: LLMDataModel, override=False):
        if self.input_data_model is not None and not override:
            raise ValueError("Input data model already registered. If you want to override it, set `override=True`.")
        self.input_data_model = data_model

    def register_output_data_model(self, data_model: LLMDataModel, override=False):
        if self.output_data_model is not None and not override:
            raise ValueError("Data model already registered. If you want to override it, set `override=True`.")
        self.output_data_model = data_model

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

    def forward(self, task: str, f_semantic_conditions: list[Callable], *args, **kwargs):
        if self.output_data_model is None:
            raise ValueError("While the input data model is optional, the output data models must be provided. Please register it before calling the `forward` method.")

        # Force JSON mode
        kwargs["response_format"] = {"type": "json_object"}
        logger.info(f"Initializing semantic validation for task '{task}' with type validated input {self.input_data_model.simplify_json_schema()} and type validated output {self.output_data_model.simplify_json_schema()}…")

        # Zero shot the task
        context = self.remedy_task.format(
            task=task,
            input_data_model=self.input_data_model.simplify_json_schema() if self.input_data_model is not None else "No input data model was provided.",
            input=self.input_data_model.print_fields() if self.input_data_model is not None else "N/A",
            output_data_model=self.output_data_model.simplify_json_schema(),
            output="N/A",
            errors="N/A",
        )
        json = super().forward(context, *args, **kwargs).value
        # Get list of seeds for remedy (to avoid same remedy for same input)
        remedy_seeds = self.prepare_seeds(self.retry_params["tries"], **kwargs)
        logger.info(f"Prepared {len(remedy_seeds)} remedy seeds for semantic validation attempts…")

        result = None
        last_error = None
        errors = []
        for i in range(self.retry_params["tries"]):
            logger.info(f"Attempt {i+1}/{self.retry_params['tries']}: Attempting semantic validation…")
            # Try to validate against provided data model
            try:
                result = self.output_data_model.model_validate_json(json, strict=True)
                if not all(f(result) for f in f_semantic_conditions):
                    raise ValueError("Semantic validation failed!")
                break
            except Exception as e:
                logger.error(f"Attempt {i+1} failed with error: {e}")

                self._pause()

                # Check if the error is pydantic
                if isinstance(e, ValidationError):
                    error_str = self.simplify_validation_errors(e)
                    logger.info(f"The following errors were identified: {error_str}")
                else:
                    error_str = str(e)
                    logger.info(f"The following error was identified: {error_str}")
                errors.append(error_str)

                # Change the dynamic context of the remedy function to account for the errors
                logger.info("Updating remedy function context…")
                context = self.remedy_task.format(
                    task=task,
                    input_data_model=self.input_data_model.simplify_json_schema() if self.input_data_model is not None else "No input data model was provided.",
                    input=self.input_data_model.print_fields() if self.input_data_model is not None else "N/A",
                    output_data_model=self.output_data_model.instruct_llm(),
                    output=json,
                    errors="\n".join(errors)
                )
                self.remedy_function.clear()
                self.remedy_function.adapt(context)
                json = self.remedy_function(seed=remedy_seeds[i]).value
                logger.info("Applied remedy function with updated context!")
                logger.info(f"New context: {self.remedy_function.dynamic_context}")

                last_error = error_str

        if result is None or not all([f(result) for f in f_semantic_conditions]):
            logger.info(f"Semantic validation attempts failed. Last error: {last_error}")
            raise SemanticValidationError(
                task=task,
                result=json,
                violations=errors,
            )

        logger.info("Semantic validation completed successfully!")
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

    @property
    def remedy_task(self):
        return (
            "You are an expert in semantic validation. Your goal is to validate the output data model given the task, and if given, the input data model, the errors, and the output that produced the errors.\n"
            "Your original task was:\n"
            "<original_task>\n"
            "{task}\n"
            "</original_task>"
            "\n\n"
            "The input data model is:\n"
            "<input_data_model>\n"
            "{input_data_model}\n"
            "</input_data_model>\n"
            "\n\n"
            "The given input was:\n"
            "<input>\n"
            "{input}\n"
            "</input>\n"
            "\n\n"
            "The output data model is:\n"
            "<output_data_model>\n"
            "{output_data_model}\n"
            "</output_data_model>"
            "\n\n"
            "You've lastly generated the following output:\n"
            "<output>"
            "{output}"
            "</output>"
            "\n\n"
            "During the semantic validation, the output was found to have the following errors:\n"
            "<errors>\n"
            "{errors}\n"
            "</errors>"
            "\n\n"
            "Your new task is to:\n"
            "1. Correct the provided output to address **all listed validation errors**.\n"
            "2. Ensure the corrected output adheres strictly to the requirements of the **original task**.\n"
            "3. Preserve the intended meaning and structure of the original task wherever possible."
            "\n\n"
            "Important guidelines:\n"
            "</guidelines>\n"
            "- Focus only on fixing the listed validation errors without introducing new errors or unnecessary changes.\n"
            "- Ensure the revised output is clear, accurate, and fully compliant with the original task.\n"
            "- Maintain proper formatting and any required conventions specified in the original task.\n"
            "</guidelines>"
        )

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
        self.pre_remedy = pre_remedy
        self.post_remedy = post_remedy
        self.retry_params = remedy_retry_params
        self.f_type_validation_remedy = TypeValidationFunction(verbose=False, **remedy_retry_params)
        self.f_semantic_validation_remedy = SemanticValidationFunction(verbose=False, **remedy_retry_params)

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

    def _validate_input(self, wrapped_self, input):
        if self.pre_remedy:
            logger.info("Validating pre-conditions with remedy...")
            if not hasattr(wrapped_self, 'pre'):
                logger.error("Pre-condition function not defined!")
                raise Exception("Pre-condition function not defined. Please define a `pre` method if you want to enforce pre-conditions through a remedy.")

            # Semantic validation with remedy if defined
            try:
                logger.info("Attempting pre-condition validation...")
                return wrapped_self.pre(input)
            except Exception as e:
                logger.error(f"Pre-condition validation failed: {str(e)}")
                if not hasattr(wrapped_self, "task"):
                    logger.error("Task attribute not defined!")
                    raise Exception("Task not defined. Please define a `task` attribute if you want to enforce semantic validation through a remedy.")
                # We don't have an input data model for the input, so we try to "fill in" the input data model so that it passes the semantic validation
                logger.info("Attempting remedy with semantic validation...")
                self.f_semantic_validation_remedy.register_output_data_model(input)
                input = self.f_semantic_validation_remedy(wrapped_self.task, f_semantic_conditions=[wrapped_self.pre])
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

    def _validate_output(self, wrapped_self, input, output):
        logger.info("Starting output validation...")
        try:
            logger.info("Registering output data model for type validation...")
            self.f_type_validation_remedy.register_data_model(output)
            output = self.f_type_validation_remedy()
            logger.info("Type validation successful")
        except Exception as e:
            logger.error(f"Type validation failed: {str(e)}")
            raise Exception("Type validation failed! Couldn't create a data model matching the output data model.")

        # Validate output data model against semantic conditions
        if self.post_remedy:
            logger.info("Validating post-conditions with remedy...")
            if not hasattr(wrapped_self, "post"):
                logger.error("Post-conditions not defined!")
                raise Exception("Post-conditions not defined. Please define a `post` attribute if you want to enforce semantic validation through a remedy.")
            if not hasattr(wrapped_self, "task"):
                logger.error("Task attribute not defined!")
                raise Exception("Task not defined. Please define a `task` attribute if you want to enforce semantic validation through a remedy.")

            logger.info("Setting up semantic validation...")
            self.f_semantic_validation_remedy.register_input_data_model(input)
            self.f_semantic_validation_remedy.register_output_data_model(output)
            output = self.f_semantic_validation_remedy(wrapped_self.task, f_semantic_conditions=[wrapped_self.post])
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
        # Store original methods
        contract_self = self
        original_init = cls.__init__
        original_forward = cls.forward

        def __init__(wrapped_self, *args, **kwargs):
            logger.info("Initializing contract...")
            original_init(wrapped_self, *args, **kwargs)
            wrapped_self.contract_successful = False
            wrapped_self.contract_result = None
            logger.info("Contract initialization complete")

        def wrapped_forward(wrapped_self, *args, **kwargs):
            logger.info("Starting contract forward pass...")
            original_input = contract_self._is_valid_input(*args, **kwargs)

            sig = inspect.signature(original_forward)
            original_output_type = sig.return_annotation
            if original_output_type == inspect._empty:
                logger.error("Missing return type annotation!")
                raise ValueError("The contract requires a return type annotation.")
            if not issubclass(original_output_type, LLMDataModel):
                logger.error(f"Invalid return type: {original_output_type}")
                raise ValueError("The return type annotation must be a subclass of `LLMDataModel`.")
            try:
                # Pre-conditions check if defined in class
                input = original_input
                maybe_new_input = contract_self._validate_input(wrapped_self, input)
                if maybe_new_input is not None:
                    input = maybe_new_input

                # Post-conditions check if defined in class
                output = self._validate_output(wrapped_self, input, original_output_type)
                wrapped_self.contract_successful = True
                wrapped_self.contract_result = output
            finally:
                # Run the original forward method
                logger.info("Executing original forward method...")
                kwargs['input'] = original_input  # Set input in kwargs instead of passing it separately
                output = original_forward(wrapped_self, *args, **kwargs)
                # If contract failed, we check if the user had some follow-up actions
                if not wrapped_self.contract_successful:
                    logger.warning("Contract validation failed, checking output type...")
                    # But we must enforce that the output matches the expected type since user can override the forward method in the process
                    if not isinstance(output, original_output_type):
                        logger.error(f"Output type mismatch: {type(output)}")
                        raise TypeError(f"Expected output to be an instance of {original_output_type}, but got {type(output)}! Forward method must return an instance of {original_output_type}!")
                else:
                    logger.success("Contract validation successful")
                    # If contract passed, we always return the successful output
                    if wrapped_self.contract_result is not None:
                        return wrapped_self.contract_result
            return output

        # Replace class methods with wrapped versions
        cls.__init__ = __init__
        cls.forward = wrapped_forward

        return cls


#TODO: this is still dependent on the old usage; refactor
@deprecated("This will be changed in upcoming releases to not depend on the old usage implementation. For usage, see `symai.components.MetadataTracker`.")
class LengthConstrainedFunction(TypeValidationFunction):
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


class BaseStrategy(TypeValidationFunction):
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
