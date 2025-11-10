import inspect
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from beartype import beartype
from loguru import logger
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .components import Function
from .models import LLMDataModel, TypeValidationError, build_dynamic_llm_datamodel
from .symbol import Expression
from .utils import UserMessage


class ValidationFunction(Function):
    """
    Base class for validation functions that share common logic for:
      • Retry attempts
      • Remedy function
      • Prepare seeds
      • Pause/backoff logic
      • Error simplification
    """
    # Have some default retry params that don't add overhead
    _default_retry_params: ClassVar[dict[str, int | float | bool]] = {
        "tries": 8,
        "delay": 0.015,
        "backoff": 1.25,
        "jitter": 0.0,
        "max_delay": 0.25,
        "graceful": False,
    }

    def __init__(
        self,
        retry_params: dict[str, int | float | bool] | None = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retry_params = {**self._default_retry_params, **(retry_params or {})}
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
        return rnd.randint(
            0, np.iinfo(np.int16).max, size=num_seeds, dtype=np.int16
        ).tolist()

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

    def _pause(self, attempt):
        base = self.retry_params['delay'] * (self.retry_params['backoff'] ** attempt)
        jit = (np.random.uniform(*self.retry_params['jitter'])
            if isinstance(self.retry_params['jitter'], tuple)
            else self.retry_params['jitter'])
        _delay = min(base + jit, self.retry_params['max_delay'])
        time.sleep(_delay)

    def remedy_prompt(self, *_args, **_kwargs):
        """
        Abstract or base remedy prompt method.
        Child classes typically override this to include additional context needed for correction.
        """
        msg = "Each child class needs its own remedy_prompt implementation."
        UserMessage(msg)
        raise NotImplementedError(msg)

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
    Performs type validation on an output, ensuring it conforms to a specified
    Pydantic data model. It can also optionally perform semantic validation
    if a user provides a callable designed to semantically validate the
    structure of the type-validated data.
    """
    def __init__(
        self,
        retry_params: dict[str, int | float | bool] = ValidationFunction._default_retry_params,
        accumulate_errors: bool = False,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, retry_params=retry_params, verbose=verbose, **kwargs)
        self.input_data_model = None
        self.output_data_model = None
        self.accumulate_errors = accumulate_errors
        self.verbose = verbose

    def register_expected_data_model(self, data_model: LLMDataModel, attach_to: str, override: bool = False):
        assert attach_to in ["input", "output"], f"Invalid attach_to value: {attach_to}; must be either 'input' or 'output'"
        if attach_to == "input":
            if self.input_data_model is not None and not override:
                msg = "There is already a data model attached to the input. If you want to override it, set `override=True`."
                UserMessage(msg)
                raise ValueError(msg)
            self.input_data_model = data_model
        elif attach_to == "output":
            if self.output_data_model is not None and not override:
                msg = "There is already a data model attached to the output. If you want to override it, set `override=True`."
                UserMessage(msg)
                raise ValueError(msg)
            self.output_data_model = data_model

    def remedy_prompt(self, prompt: str, output: str, errors: str) -> str:
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
- Ensure the revised output is clear, accurate, and fully compliant with the original prompt.
- Maintain proper formatting and any required conventions specified in the original prompt.
</guidelines>
"""

    def _ensure_output_model(self):
        if self.output_data_model is None:
            msg = (
                "While the input data model is optional, the output data model must be provided. "
                "Please register it before calling the `forward` method."
            )
            UserMessage(msg)
            raise ValueError(msg)

    def _display_verbose_panels(self, prompt: str):
        if not self.verbose:
            return
        for label, body in [
            ("Prompt", prompt),
            ("Input data model", self.input_data_model.simplify_json_schema() if self.input_data_model else 'N/A'),
            ("Output data model", self.output_data_model.simplify_json_schema()),
        ]:
            self.display_panel(body, title=label)

    def _check_semantic_conditions(
        self,
        result,
        f_semantic_conditions: list[Callable] | None,
    ) -> str | None:
        if f_semantic_conditions is None:
            return None
        try:
            assert all(
                f(result if not getattr(self.output_data_model, '_is_dynamic_model', False) else result.value)
                for f in f_semantic_conditions
            )
        except Exception as err:
            return f"Semantic validation failed with:\n{err!s}"
        return None

    def _format_validation_error(self, error: Exception) -> str:
        if isinstance(error, ValidationError):
            return self.simplify_validation_errors(error)
        return str(error)

    def _handle_failed_validation_attempt(
        self,
        attempt_index: int,
        prompt: str,
        json_str: str,
        errors: list[str],
        error: Exception,
        remedy_seeds: list[Any],
        kwargs: dict,
    ) -> str:
        logger.info(f"Validation attempt {attempt_index + 1} failed, pausing before retry…")
        self._pause(attempt_index)
        error_str = self._format_validation_error(error)
        errors.append(error_str)

        logger.error("Validation errors identified!")
        if self.verbose:
            errors_report = "\n".join(errors) if self.accumulate_errors else error_str
            title = f"Validation Errors ({'accumulated errors' if self.accumulate_errors else 'last error'})"
            self.display_panel(errors_report, title=title, border_style="red")

        logger.info("Updating remedy function context…")
        context = self.remedy_prompt(
            prompt=prompt,
            output=json_str,
            errors="\n".join(errors) if self.accumulate_errors else error_str,
        )
        self.remedy_function.clear()
        self.remedy_function.adapt(context)
        if self.verbose:
            self.display_panel(self.remedy_function.dynamic_context, title="New Context")

        json_str = self.remedy_function(seed=remedy_seeds[attempt_index], **kwargs).value
        logger.info("Applied remedy function with updated context!")
        return json_str

    def _run_validation_attempts(
        self,
        prompt: str,
        f_semantic_conditions: list[Callable] | None,
        validation_context: dict,
        remedy_seeds: list[Any],
        json_str: str,
        kwargs: dict,
    ) -> tuple[Any | None, str, list[str]]:
        errors: list[str] = []
        result = None
        total_attempts = self.retry_params["tries"] + 1
        for attempt in range(total_attempts):
            if attempt != self.retry_params["tries"]:
                logger.info(f"Attempt {attempt + 1}/{self.retry_params['tries']}: Attempting validation…")
            try:
                result = self.output_data_model.model_validate_json(
                    json_str,
                    strict=False,
                    context=validation_context,
                )
                semantic_error = self._check_semantic_conditions(result, f_semantic_conditions)
                if semantic_error is not None:
                    if attempt == total_attempts - 1:
                        result = None
                        errors.append(semantic_error)
                        break
                    raise AssertionError(semantic_error)
                break
            except Exception as error:
                json_str = self._handle_failed_validation_attempt(
                    attempt,
                    prompt,
                    json_str,
                    errors,
                    error,
                    remedy_seeds,
                    kwargs,
                )
        return result, json_str, errors

    def _handle_validation_failure(self, prompt: str, json_str: str, errors: list[str]):
        logger.error("All validation attempts failed!")
        if self.retry_params['graceful']:
            return
        raise TypeValidationError(
            prompt=prompt,
            result=json_str,
            violations=errors,
        )

    def forward(self, prompt: str, f_semantic_conditions: list[Callable] | None = None, *args, **kwargs):
        self._ensure_output_model()
        validation_context = kwargs.pop('validation_context', {})
        kwargs["response_format"] = {"type": "json_object"}
        logger.info("Initializing validation…")
        self._display_verbose_panels(prompt)

        context = self.zero_shot_prompt(prompt=prompt)
        json_str = super().forward(context, *args, **kwargs).value

        remedy_seeds = self.prepare_seeds(self.retry_params["tries"] + 1, **kwargs)
        logger.info(f"Prepared {len(remedy_seeds)} remedy seeds for validation attempts…")

        result, json_str, errors = self._run_validation_attempts(
            prompt,
            f_semantic_conditions,
            validation_context,
            remedy_seeds,
            json_str,
            kwargs,
        )

        if result is None:
            return self._handle_validation_failure(prompt, json_str, errors)

        logger.success("Validation completed successfully!")
        self.remedy_function.clear()
        return result


@beartype
class contract:
    _default_remedy_retry_params: ClassVar[dict[str, int | float | bool]] = {
        "tries": 8,
        "delay": 0.015,
        "backoff": 1.25,
        "jitter": 0.0,
        "max_delay": 0.25,
        "graceful": False,
    }
    _internal_forward_kwargs: ClassVar[set[str]] = {"validation_context"}

    def __init__(
        self,
        pre_remedy: bool = False,
        post_remedy: bool = False,
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
        self.remedy_retry_params = remedy_retry_params
        self.f_type_validation_remedy = TypeValidationFunction(accumulate_errors=accumulate_errors, verbose=verbose, retry_params=remedy_retry_params)

        if not verbose:
            logger.disable(__name__)
        else:
            logger.enable(__name__)

    def _is_valid_input(self, input_value):
        if input_value is None:
            logger.error("No `input` argument provided!")
            msg = "Please provide an `input` argument."
            UserMessage(msg)
            raise ValueError(msg)
        if not isinstance(input_value, LLMDataModel):
            logger.error(f"Invalid input type: {type(input_value)}")
            msg = f"Expected input to be of type `LLMDataModel`, got {type(input_value)}"
            UserMessage(msg)
            raise TypeError(msg)
        return True

    def _is_valid_output(self, output_type):
        if output_type == inspect._empty:
            logger.error("Missing return type annotation!")
            msg = "The contract requires a return type annotation."
            UserMessage(msg)
            raise ValueError(msg)
        if not issubclass(output_type, LLMDataModel):
            logger.error(f"Invalid return type: {output_type}")
            msg = "The return type annotation must be a subclass of `LLMDataModel`."
            UserMessage(msg)
            raise TypeError(msg)
        return True

    def _try_dynamic_type_annotation(self, original_forward, *, context: str = "input"):
        assert context in {"input", "output"}, (
            "`context` must be either 'input' or 'output', got " + repr(context)
        )
        sig = inspect.signature(original_forward)
        try:
            resolved_param = None
            # Fallback: look at the relevant part of the function signature
            # depending on whether we deal with an *input* or *output*
            if context == "input":
                param = sig.parameters.get("input")
                if param is None:
                    for candidate in sig.parameters.values():
                        if candidate.name == "self":
                            continue
                        if candidate.kind in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        ):
                            param = candidate
                            break
                resolved_param = param
                if param is None or param.annotation == inspect._empty:
                    msg = "Failed to infer type from input parameter annotation"
                    UserMessage(msg)
                    raise TypeError(msg)
                dynamic_model = build_dynamic_llm_datamodel(param.annotation)
            else:  # context == "output"
                if sig.return_annotation == inspect._empty:
                    msg = "Failed to infer type from return annotation"
                    UserMessage(msg)
                    raise TypeError(msg)
                dynamic_model = build_dynamic_llm_datamodel(sig.return_annotation)
        except Exception as err:
            logger.exception(f"Failed to build dynamic LLMDataModel from {resolved_param}!")
            msg = (
                "The type annotation must be a subclass of `LLMDataModel` or a "
                "valid Python typing object supported by Pydantic."
            )
            UserMessage(msg)
            raise TypeError(msg) from err

        dynamic_model._is_dynamic_model = True
        return dynamic_model

    def _try_remedy_with_exception(self, prompt, f_semantic_conditions, **remedy_kwargs):
        try:
            data_model = self.f_type_validation_remedy(prompt, f_semantic_conditions=f_semantic_conditions, **remedy_kwargs)
        except Exception as e:
            logger.error("Type validation failed with exception!")
            raise e
        return data_model

    def _validate_input(self, wrapped_self, input_value, it, **remedy_kwargs):
        logger.info("Starting input validation...")
        if self.pre_remedy:
            logger.info("Validating pre-conditions with remedy...")
            if not hasattr(wrapped_self, 'pre'):
                logger.error("Pre-condition function not defined!")
                msg = "Pre-condition function not defined. Please define a `pre` method if you want to enforce pre-conditions through a remedy."
                UserMessage(msg)
                raise Exception(msg)

            op_start = time.perf_counter()
            try:
                assert wrapped_self.pre(input_value)
                logger.success("Pre-condition validation successful!")
                return input_value
            except Exception:
                logger.exception("Pre-condition validation failed!")
                self.f_type_validation_remedy.register_expected_data_model(input_value, attach_to="output", override=True)
                input_value = self._try_remedy_with_exception(
                    prompt=wrapped_self.prompt,
                    f_semantic_conditions=[wrapped_self.pre],
                    **remedy_kwargs,
                )
            finally:
                wrapped_self._contract_timing[it]["input_validation"] = time.perf_counter() - op_start
            return input_value
        if hasattr(wrapped_self, 'pre'):
            logger.info("Validating pre-conditions without remedy...")
            op_start = time.perf_counter()
            try:
                assert wrapped_self.pre(input_value)
            except Exception as e:
                logger.exception("Pre-condition validation failed")
                raise e
            finally:
                wrapped_self._contract_timing[it]["input_validation"] = time.perf_counter() - op_start
            logger.success("Pre-condition validation successful!")
            return input_value
        logger.info("Skip; no pre-condition validation was required!")
        return input_value

    def _validate_output(self, wrapped_self, input_value, output, it, **remedy_kwargs):
        logger.info("Starting output validation...")
        self.f_type_validation_remedy.register_expected_data_model(input_value, attach_to="input", override=True)
        self.f_type_validation_remedy.register_expected_data_model(output, attach_to="output", override=True)

        op_start = time.perf_counter()
        try:
            logger.info("Getting a valid output type...")
            output = self._try_remedy_with_exception(prompt=wrapped_self.prompt, f_semantic_conditions=None, **remedy_kwargs)
            if output is None: # output is None when graceful mode is enabled
                return output
        except Exception as e:
            logger.exception("Type creation failed!")
            raise e
        finally:
            wrapped_self._contract_timing[it]["output_validation"] = time.perf_counter() - op_start
        logger.success("Type successfully created!")

        if self.post_remedy:
            logger.info("Validating post-conditions with remedy...")
            if not hasattr(wrapped_self, "post"):
                logger.error("Post-condition function not defined!")
                msg = "Post-condition function not defined. Please define a `post` method if you want to enforce post-conditions through a remedy."
                UserMessage(msg)
                raise Exception(msg)

            op_start = time.perf_counter()
            try:
                assert wrapped_self.post(output)
                logger.success("Post-condition validation successful!")
                return output
            except Exception:
                logger.exception("Post-condition validation failed!")
                output = self._try_remedy_with_exception(prompt=wrapped_self.prompt, f_semantic_conditions=[wrapped_self.post], **remedy_kwargs)
            finally:
                wrapped_self._contract_timing[it]["output_validation"] += (time.perf_counter() - op_start)
            logger.success("Post-condition validation successful!")
            return output
        if hasattr(wrapped_self, "post"):
            logger.info("Validating post-conditions without remedy...")
            op_start = time.perf_counter()
            try:
                assert wrapped_self.post(output)
            except Exception as e:
                logger.exception("Post-condition validation failed!")
                raise e
            finally:
                wrapped_self._contract_timing[it]["output_validation"] = time.perf_counter() - op_start
            logger.success("Post-condition validation successful!")
            return output
        logger.info("Skip; no post-condition validation was required!")
        return output

    def _validate_act_method(self, act_method):
        act_sig = inspect.signature(act_method)
        params = list(act_sig.parameters.values())

        first_param = None
        for param in params:
            if param.name == "self":
                continue
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                first_param = param
                break

        if first_param is None:
            msg = "'act' method must accept at least one positional parameter after `self`."
            UserMessage(msg)
            raise TypeError(msg)
        if first_param.annotation == inspect._empty:
            msg = f"'act' method parameter '{first_param.name}' must have a type annotation."
            UserMessage(msg)
            raise TypeError(msg)
        if act_sig.return_annotation == inspect._empty:
            msg = "'act' method must have a return type annotation'"
            UserMessage(msg)
            raise TypeError(msg)
        return True

    def _act(self, wrapped_self, input_value, it, **act_kwargs):
        act_method = getattr(wrapped_self, 'act', None)
        if not callable(act_method):
            # Propagate the input if no act method is defined
            return input_value

        assert self._validate_act_method(act_method)

        is_dynamic_model = getattr(input_value, '_is_dynamic_model', False)
        input_value = input_value if not is_dynamic_model else input_value.value

        logger.info(f"Executing 'act' method on {wrapped_self.__class__.__name__}…")

        op_start = time.perf_counter()
        try:
            act_output = act_method(input_value, **act_kwargs)
        except Exception as e:
            logger.exception("'act' method execution failed")
            raise e
        finally:
            wrapped_self._contract_timing[it]["act_execution"] = time.perf_counter() - op_start

        act_sig = inspect.signature(act_method)
        if (
            act_sig.return_annotation != inspect.Signature.empty
            and inspect.isclass(act_sig.return_annotation)
            and not isinstance(act_output, act_sig.return_annotation)
        ):
            msg = f"'act' method returned {type(act_output).__name__}, expected {act_sig.return_annotation.__name__}."
            UserMessage(msg)
            raise TypeError(msg)

        logger.success("'act' method executed successfully!")
        return act_output

    def _build_wrapped_init(self, original_init):
        def __init__(wrapped_self, *args, **kwargs):
            logger.info("Initializing contract...")
            original_init(wrapped_self, *args, **kwargs)

            if not hasattr(wrapped_self, "prompt"):
                logger.error("Prompt attribute not defined!")
                msg = "Please define a static `prompt` attribute that describes what the contract must do."
                UserMessage(msg)
                raise Exception(msg)

            wrapped_self.contract_successful = False
            wrapped_self.contract_result = None
            wrapped_self.contract_exception = None
            wrapped_self._contract_timing = defaultdict(dict)
            logger.info("Contract initialization complete!")

        return __init__

    def _start_contract_execution(self, wrapped_self):
        logger.info("Starting contract execution...")
        it = len(wrapped_self._contract_timing)  # the len is the __call__ op_start
        return it, time.perf_counter()

    def _find_input_param_name(self, sig: inspect.Signature) -> str | None:
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return param.name
        return None

    def _prepare_forward_args(self, args, kwargs):
        args_list = list(args)
        kwargs_without_input = dict(kwargs)
        original_kwargs = dict(kwargs)
        return args_list, kwargs_without_input, original_kwargs

    def _extract_input_value(
        self,
        args_list,
        kwargs_without_input,
        original_kwargs,
        input_param_name: str | None,
    ):
        if args_list:
            return args_list[0], ("args", 0)
        if input_param_name and input_param_name in kwargs_without_input:
            return kwargs_without_input.pop(input_param_name), ("kwargs", input_param_name)
        if "input" in kwargs_without_input:
            return kwargs_without_input.pop("input"), ("kwargs", "input")
        return original_kwargs.get("input"), ("fallback_kw", "input")

    def _coerce_input_value(self, original_forward, input_value):
        try:
            assert self._is_valid_input(input_value)
            return input_value
        except TypeError:
            input_type_model = self._try_dynamic_type_annotation(original_forward, context="input")
            return input_type_model(value=input_value)

    def _collect_validation_kwargs(self, wrapped_self, kwargs_without_input):
        maybe_payload = getattr(wrapped_self, "payload", None)
        maybe_template = getattr(wrapped_self, "template", None)
        if inspect.ismethod(maybe_template):
            maybe_template = None
        return {
            **kwargs_without_input,
            "payload": maybe_payload,
            "template_suffix": maybe_template,
        }

    def _resolve_output_type(self, sig: inspect.Signature, original_forward):
        output_type = sig.return_annotation
        try:
            assert self._is_valid_output(output_type)
        except TypeError:
            output_type = self._try_dynamic_type_annotation(original_forward, context="output")
        return output_type

    def _run_contract_pipeline(
        self,
        wrapped_self,
        current_input_value,
        output_type,
        it,
        validation_kwargs,
    ):
        output = None
        try:
            maybe_new_input = self._validate_input(wrapped_self, current_input_value, it, **validation_kwargs)
            if maybe_new_input is not None:
                current_input_value = maybe_new_input

            current_input_value = self._act(wrapped_self, current_input_value, it, **validation_kwargs)

            output = self._validate_output(
                wrapped_self,
                current_input_value,
                output_type,
                it,
                **validation_kwargs,
            )
            wrapped_self.contract_successful = output is not None
            wrapped_self.contract_result = output
            wrapped_self.contract_exception = None
        except Exception as exc:
            logger.exception("Contract execution failed in main path!")
            wrapped_self.contract_successful = False
            wrapped_self.contract_exception = exc
        return output, current_input_value

    def _execute_forward_call(
        self,
        wrapped_self,
        original_forward,
        args_list,
        original_kwargs,
        input_param_name,
        input_source,
        forward_input_value,
        it,
        contract_start,
    ):
        forward_kwargs = original_kwargs.copy()
        for internal_kw in self._internal_forward_kwargs:
            forward_kwargs.pop(internal_kw, None)

        logger.info("Executing original forward method...")

        if input_param_name:
            if input_param_name in forward_kwargs or input_source == ("kwargs", input_param_name):
                forward_kwargs[input_param_name] = forward_input_value
            elif input_source == ("args", 0) and args_list:
                args_list[0] = forward_input_value
            else:
                forward_kwargs[input_param_name] = forward_input_value
        else:
            forward_kwargs['input'] = forward_input_value

        if input_param_name and input_param_name != "input" and "input" in forward_kwargs:
            forward_kwargs.pop("input")

        try:
            op_start = time.perf_counter()
            output = original_forward(wrapped_self, *args_list, **forward_kwargs)
        finally:
            wrapped_self._contract_timing[it]["forward_execution"] = time.perf_counter() - op_start
        wrapped_self._contract_timing[it]["contract_execution"] = time.perf_counter() - contract_start
        return output

    def _finalize_contract_output(self, output, output_type, wrapped_self):
        if not isinstance(output, output_type):
            logger.error(f"Output type mismatch: {type(output)}")
            if self.remedy_retry_params["graceful"]:
                if getattr(output_type, '_is_dynamic_model', False) and hasattr(output, 'value'):
                    return output.value
                return output
            msg = (
                f"Expected output to be an instance of {output_type}, "
                f"but got {type(output)}! Forward method must return an instance of {output_type}!"
            )
            UserMessage(msg)
            raise TypeError(msg)
        if not wrapped_self.contract_successful:
            logger.warning("Contract validation failed!")
        else:
            logger.success("Contract validation successful!")

        if getattr(output_type, '_is_dynamic_model', False):
            return output.value
        return output

    def _contract_forward_impl(self, wrapped_self, original_forward, *args, **kwargs):
        it, contract_start = self._start_contract_execution(wrapped_self)
        sig = inspect.signature(original_forward)
        input_param_name = self._find_input_param_name(sig)
        args_list, kwargs_without_input, original_kwargs = self._prepare_forward_args(args, kwargs)
        input_value, input_source = self._extract_input_value(args_list, kwargs_without_input, original_kwargs, input_param_name)
        current_input_value = self._coerce_input_value(original_forward, input_value)
        input_value = current_input_value
        validation_kwargs = self._collect_validation_kwargs(wrapped_self, kwargs_without_input)
        output_type = self._resolve_output_type(sig, original_forward)

        output, current_input_value = self._run_contract_pipeline(
            wrapped_self,
            current_input_value,
            output_type,
            it,
            validation_kwargs,
        )

        forward_input_value = current_input_value if wrapped_self.contract_successful else input_value
        output = self._execute_forward_call(
            wrapped_self,
            original_forward,
            args_list,
            original_kwargs,
            input_param_name,
            input_source,
            forward_input_value,
            it,
            contract_start,
        )

        return self._finalize_contract_output(output, output_type, wrapped_self)

    def _build_wrapped_forward(self, original_forward):
        def wrapped_forward(wrapped_self, *args, **kwargs):
            return self._contract_forward_impl(wrapped_self, original_forward, *args, **kwargs)

        return wrapped_forward

    def _build_contract_perf_stats(self):
        def contract_perf_stats(wrapped_self):
            """Analyzes and prints timing statistics across all forward calls."""
            console = Console()

            num_calls = len(wrapped_self._contract_timing)
            if num_calls == 0:
                console.print("No contract executions recorded.")
                return {}

            ordered_operations = [
                "input_validation",
                "act_execution",
                "output_validation",
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

            sum_tracked_times = sum(stats[op]['total'] for op in ordered_operations[:-1])
            overhead_time = total_execution_time - sum_tracked_times
            overhead_percentage = (overhead_time / total_execution_time) * 100 if total_execution_time > 0 else 0

            stats['overhead'] = {
                'count': num_calls,
                'total': overhead_time,
                'mean': overhead_time / num_calls if num_calls > 0 else 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'percentage': overhead_percentage
            }

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

            s = stats['overhead']
            table.add_row(
                "Overhead",
                str(s['count']),
                f"{s['total']:.3f}",
                f"{s['mean']:.3f}",
                f"{s['std']:.3f}",
                f"{s['min']:.3f}",
                f"{s['max']:.3f}",
                f"{s['percentage']:.1f}%",
                style="bold blue"
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

        return contract_perf_stats

    def __call__(self, cls):
        original_init = cls.__init__
        original_forward = cls.forward

        cls.__init__ = self._build_wrapped_init(original_init)
        cls.forward = self._build_wrapped_forward(original_forward)
        cls.contract_perf_stats = self._build_contract_perf_stats()
        return cls


class BaseStrategy(TypeValidationFunction):
    def __init__(self, data_model: BaseModel, *_args, **kwargs):
        super().__init__(
            retry_params={
                "tries": 8,
                "delay": 0.015,
                "backoff": 1.25,
                "jitter": 0.0,
                "max_delay": 0.25,
                "graceful": False,
            },
            **kwargs,
        )
        super().register_expected_data_model(data_model, attach_to="output")
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def forward(self, *args, **kwargs):
        return super().forward(
            *args,
            payload=self.payload,
            template_suffix=self.template,
            response_format={"type": "json_object"},
            **kwargs,
        )

    @property
    def payload(self):
        return None

    @property
    def static_context(self):
        raise NotImplementedError

    @property
    def template(self):
        return "{{fill}}"


class Strategy(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def __new__(cls, module: str, *_args, **_kwargs):
        cls._module = module
        cls.module_path = 'symai.extended.strategies'
        return Strategy.load_module_class(cls.module)
