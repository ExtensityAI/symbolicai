# Ensuring Correctness with Contracts

In SymbolicAI, the `@contract` decorator provides a powerful mechanism, inspired by Design by Contract (DbC) principles, to enhance the reliability and semantic correctness of `Expression` classes, especially those interacting with Large Language Models (LLMs). It allows you to define explicit pre-conditions, post-conditions, and intermediate processing steps, guiding the behavior of your classes and the underlying LLMs. The original post introducing this feature can be found [here](https://futurisold.github.io/2025-03-01-dbc/).

## Why Use Contracts?

Traditional software development often relies on testing to verify correctness after the fact. Contracts, however, encourage building correctness into the design itself. When working with LLMs, which are inherently probabilistic, ensuring that outputs are not only syntactically valid but also semantically meaningful and contextually appropriate is crucial.

Contracts in SymbolicAI help bridge this gap by:

1.  **Enforcing Semantic Guarantees**: Beyond static type checking (which ensures structural validity), contracts allow you to define and validate what your `Expression`'s inputs and outputs *mean* in a given context.
2.  **Guiding LLM Behavior**: The error messages raised by failed pre-conditions and post-conditions are used as corrective prompts, enabling the LLM to attempt self-correction. This turns validation failures into learning opportunities for the model.
3.  **Proactive Structuring**: Designing a contract forces careful consideration of inputs, outputs, and invariants, shifting from reactive validation to proactive structuring of your logic.
4.  **Improving Predictability and Reliability**: By setting clear expectations and validation steps, contracts make your AI components more predictable and less prone to unexpected or undesirable outputs (like hallucinations).
5.  **Enhancing Composability**: Clear contracts at the interface level allow different components (potentially powered by different LLMs or even rule-based systems) to interoperate reliably, as long as they satisfy the agreed-upon contractual obligations.

## What is a `@contract` in SymbolicAI?

The `@contract` is a class decorator that you apply to your custom classes inheriting from `symai.Expression`. It augments your class, particularly its `forward` method, by wrapping it with a validation and execution pipeline.

Key characteristics:

*   **Operates on `LLMDataModel`**: Inputs to and outputs from the core contract-validated logic *must* be instances of `symai.models.LLMDataModel` (which extends Pydantic's `BaseModel`). This allows for structured data validation and rich schema descriptions that can inform LLM prompts.
*   **User-Defined Conditions**: You define the contract's terms by implementing specific methods: `pre` (pre-conditions), `act` (optional intermediate action), and `post` (post-conditions), along with a `prompt` property.
*   **Fallback Mechanism**: A contract never entirely prevents the execution of your class's original `forward` method. If contract validation fails (even after remedies), your `forward` method is still called (typically with the original, unvalidated input, if the failure happened before `act`, or the `act`-modified input if failure was in `post`), allowing you to implement fallback logic or return a default, type-compliant object.
*   **State and Results**: The decorator adds attributes to your class instance:
    *   `self.contract_successful` (bool): Indicates if all contract validations (including remedies) passed.
    *   `self.contract_result` (Any): Holds the validated and potentially remedied output if successful; otherwise, it might be `None` or an intermediate value if an error occurred before `_validate_output` completed successfully.
    *   `self.contract_perf_stats()` (method): Returns a dictionary with performance metrics for various stages of the contract execution.

## Core Components of a Contracted Class

To use the `@contract` decorator, you'll define several key components within your `Expression` subclass:

### 1. The `@contract` Decorator

Apply it directly above your class definition:

```python
from symai import Expression
from symai.strategy import contract
from symai.models import LLMDataModel
from typing import Optional, List # For type hints in examples

# Default retry parameters used if not overridden in the decorator call
DEFAULT_RETRY_PARAMS = {
    "tries": 5, "delay": 0.5, "max_delay": 15,
    "jitter": 0.1, "backoff": 2, "graceful": False
}

@contract(
    pre_remedy: bool = False,
    post_remedy: bool = True,
    accumulate_errors: bool = False,
    verbose: bool = False,
    remedy_retry_params: dict = DEFAULT_RETRY_PARAMS # Uses defined defaults
)
class MyContractedClass(Expression):
    # ... class implementation ...
    pass
```

**Decorator Parameters and Defaults**:

*   `pre_remedy` (bool, default: `False`):
    If `True`, attempts to automatically correct input validation failures (from your `pre` method) using LLM-based semantic remediation.
*   `post_remedy` (bool, default: `True`):
    If `True`, attempts to automatically correct output validation failures (from your `post` method or type mismatches) using LLM-based type and semantic remediation.
*   `accumulate_errors` (bool, default: `False`):
    Controls whether error messages from multiple failed validation attempts (during remediation) are accumulated and provided to the LLM in subsequent retry attempts. See more details in the "Error Accumulation" section below.
*   `verbose` (bool, default: `False`):
    If `True`, enables detailed logging of the contract's internal operations, including prompts sent to the LLM and validation steps.
*   `remedy_retry_params` (dict, default: `{ "tries": 5, "delay": 0.5, "max_delay": 15, "jitter": 0.1, "backoff": 2, "graceful": False }`):
    A dictionary configuring the retry behavior for both type and semantic validation/remediation functions.
    *   `tries` (int): Maximum number of retry attempts for a failed validation.
    *   `delay` (float): Initial delay (in seconds) before the first retry.
    *   `max_delay` (float): The maximum delay between retries.
    *   `jitter` (float): A factor for adding random jitter to delays to prevent thundering herd problems.
    *   `backoff` (float): The multiplier for increasing the delay between retries (e.g., 2 means delay doubles).
    *   `graceful` (bool): If `True`, suppresses exceptions during retry exhaustion and might allow the process to continue with a potentially invalid state (behavior depends on the specific validation function). Typically `False` for contracts to ensure failures are robustly handled.

### 2. Input and Output Data Models

Your contract's core logic (especially `pre`, `act`, `post`, and `forward`) will operate on instances of `LLMDataModel`. Define these models using Pydantic syntax. Crucially, use `Field(description="...")` for your model attributes, as these descriptions are used to generate more effective prompts for the LLM.

```python
from pydantic import Field

class MyInput(LLMDataModel):
    text: str = Field(description="The input text to be processed.")
    max_length: Optional[int] = Field(default=None, description="Optional maximum length for processing.")

class MyIntermediate(LLMDataModel):
    processed_text: str = Field(description="Text after initial processing by 'act'.")
    entities_found: List[str] = Field(default_factory=list, description="Entities identified in 'act'.")

class MyOutput(LLMDataModel):
    result: str = Field(description="The final processed result.")
    is_valid: bool = Field(description="Indicates if the result is considered valid by post-conditions.")
```

### 3. The `prompt` Property

Your class must define a `prompt` property that returns a string. This prompt provides the high-level instructions or context to the LLM for the main task your class is designed to perform. It's particularly used by `SemanticValidationFunction` during the input (`pre_remedy`) and output (`post_remedy`) validation and remediation phases.

```python
    @property
    def prompt(self) -> str:
        return "You are an expert assistant. Given the input text, process it and return a concise summary."
```

### Error Accumulation (`accumulate_errors`)

The `accumulate_errors` parameter (default: `False`) in the `@contract` decorator influences how the underlying validation and remediation functions (`TypeValidationFunction` and `SemanticValidationFunction`) handle repeated failures during the remedy process.

*   **When `accumulate_errors = True`**:
    If a validation (e.g., a `post`-condition) fails, and a remedy attempt also fails, the error message from this failed remedy attempt is stored. If subsequent remedy attempts also fail, their error messages are appended to the list of previous errors. This accumulated list of errors is then provided as part of the context to the LLM in the next retry.
    *   **Benefits**: This can be very useful in complex scenarios. By seeing the history of what it tried and why those attempts were flagged as incorrect, the LLM might gain a better understanding of the constraints and be less likely to repeat the same mistakes. It's like showing the LLM its "thought process" and where it went wrong, potentially leading to more effective self-correction. This is particularly helpful if an initial fix inadvertently introduces a new problem that was previously not an issue, or if a previously fixed error reappears.
    *   **Potential Downsides**: In some cases, providing a long list of past errors (especially if they are somewhat contradictory or if the LLM fixed an issue that then reappears in the error list) could confuse the LLM. It might lead to an overly complex prompt that makes it harder for the model to focus on the most recent or critical issue.

*   **When `accumulate_errors = False` (Default)**:
    Only the error message from the most recent failed validation/remedy attempt is provided to the LLM for the next retry. The history of previous errors is not explicitly passed.
    *   **Benefits**: This keeps the corrective prompt focused and simpler, potentially being more effective for straightforward errors where historical context isn't necessary or could be distracting.
    *   **Potential Downsides**: The LLM loses the context of previous failed attempts. It might retry solutions that were already found to be problematic or might reintroduce errors that it had previously fixed in an earlier iteration of the remedy loop for the same overall validation step.

Choosing whether to enable `accumulate_errors` depends on the complexity of your validation logic and how you observe the LLM behaving during remediation. If you find the LLM cycling through similar errors or reintroducing past mistakes, setting `accumulate_errors=True` might be beneficial. If the remediation prompts become too noisy or confusing, `False` might be preferable.

### 4. The `pre(self, input: MyInput) -> bool` Method

This method defines the pre-conditions for your contract. It's called with the validated input object (`current_input` in `strategy.py`, which has already passed the `_is_valid_input` type check).

*   **Signature**: `def pre(self, input: YourInputModel) -> bool:`
*   **Behavior**:
    *   If all pre-conditions are met, it should do nothing or simply `return True`. (Note: The `bool` return type is conventional; the primary success signal is the absence of an exception).
    *   If a pre-condition is violated, it **must raise an exception**. The exception's message should be descriptive, as it will be used to guide the LLM if `pre_remedy` is enabled.

```python
    def pre(self, input: MyInput) -> bool:
        if not input.text:
            raise ValueError("Input text cannot be empty. Please provide some text to process.")
        if input.max_length is not None and len(input.text) > input.max_length:
            raise ValueError(f"Input text exceeds maximum length of {input.max_length}. Please provide shorter text.")
        return True
```

### 5. The `act(self, input: MyInput, **kwargs) -> MyIntermediate` Method (Optional)

The `act` method provides an optional intermediate processing step that occurs *after* input pre-validation (and potential pre-remedy) and *before* the main output validation/generation phase (`_validate_output`).

*   **Signature**: `def act(self, input: YourInputModelOrActInputModel, **kwargs) -> YourIntermediateModel:`
    *   The `input` parameter **must be named `input`** and be type-hinted with an `LLMDataModel` subclass.
    *   It must have a return type annotation, also an `LLMDataModel` subclass. This can be a different type than the input, allowing `act` to transform the data.
    *   `**kwargs` from the original call (excluding `'input'`) are passed to `act`.
*   **Behavior**:
    *   Perform transformations on the `input`, computations, or state updates on `self`.
    *   The object returned by `act` becomes the `current_input` for the `_validate_output` stage (where the LLM is typically called to generate the final output type).
    *   Can modify `self` (e.g., update instance counters, accumulate history).

```python
    # Example: Add a counter to the class for state mutation
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.calls_count = 0

    def act(self, input: MyInput, **kwargs) -> MyIntermediate:
        # self.calls_count += 1 # Example of state mutation
        processed_text = input.text.strip().lower()
        # Example: simple entity "extraction"
        entities = [word for word in processed_text.split() if len(word) > 5]
        return MyIntermediate(processed_text=processed_text, entities_found=entities)
```

### 6. The `post(self, output: MyOutput) -> bool` Method

This method defines the post-conditions. It's called by `_validate_output` with an instance of the `forward` method's declared return type (e.g., `MyOutput`). This instance is typically generated by an LLM call within `_validate_output` based on your class's `prompt` and the (potentially `act`-modified) input.

*   **Signature**: `def post(self, output: YourOutputModel) -> bool:`
*   **Behavior**:
    *   If all post-conditions are met, `return True`.
    *   If a post-condition is violated, **raise an exception** with a descriptive message. This message guides LLM self-correction if `post_remedy` is enabled.

```python
    def post(self, output: MyOutput) -> bool:
        if not output.result:
            raise ValueError("The final result string cannot be empty.")
        if output.is_valid is False and len(output.result) < 10:
            raise ValueError("If result is marked invalid, it should at least have a short explanation (min 10 chars).")
        # Example: could use self.custom_threshold modified by act
        # if hasattr(self, 'custom_threshold') and output.some_score < self.custom_threshold:
        #     raise ValueError("Score too low based on dynamic threshold.")
        return True
```

### 7. The `forward(self, input: MyInput, **kwargs) -> MyOutput` Method

This is your class's original `forward` method, containing the primary logic. The `@contract` decorator wraps this method.

*   **Signature**: `def forward(self, input: YourInputModel, **kwargs) -> YourOutputModel:`
    *   The `input` parameter **must be named `input`** and be type-hinted with an `LLMDataModel` subclass that matches (or is compatible with) the input to `pre` and `act`.
    *   It **must have a return type annotation** (e.g., `-> YourOutputModel`), which must be an `LLMDataModel` subclass. This declared type is crucial for the contract's type validation and output generation phases.
    *   It **must not use positional arguments (`*args`)**; only keyword arguments are supported for the main input. Other `**kwargs` are passed through to the neurosymbolic engine.
*   **Behavior**:
    *   This method is **always called** by the contract's `wrapped_forward` (in its `finally` block), regardless of whether the preceding contract validations (`pre`, `act`, `post`, remedies) succeeded or failed.
    *   **Developer Responsibility**: Inside your `forward` method, you *must* check `self.contract_successful` and/or `self.contract_result`.
        *   If `self.contract_successful` is `True`, `self.contract_result` holds the validated (and possibly remedied) output from the contract pipeline. You should typically return this.
        *   If `self.contract_successful` is `False`, the contract failed. `self.contract_result` might be `None` or an intermediate (invalid) object. In this case, your `forward` method should implement fallback logic:
            *   Return a sensible default object that matches `YourOutputModel`.
            *   Or, if appropriate, raise a custom exception (though the pattern encourages graceful fallback).
    *   The `input` argument received by *this* `forward` method (the one you write) depends on whether the contract succeeded:
        *   If `contract_successful == True`: `input` is the `current_input` from `wrapped_forward` which was used by `_validate_output`. This `current_input` is the output of `_act` if `act` is defined, otherwise it's the output of `_validate_input`.
        *   If `contract_successful == False`: `input` is the `original_input` (the raw input to the contract call, after initial type validation by `_is_valid_input` but before `pre` or `act` modifications).

```python
    # Ensure a logger is configured and available if you uncomment logging lines,
    # e.g., by importing from symai.utils.logger or using your application's logger.
    # import logging
    # logger = logging.getLogger(__name__) # Or self.logger if set in __init__

    def forward(self, input: MyInput, **kwargs) -> MyOutput:
        # Example: if 'name' attribute exists on the class and a logger is available
        # logger.info(f"Original forward in {getattr(self, 'name', self.__class__.__name__)} called.")

        if not self.contract_successful or self.contract_result is None:
            # Contract failed, or result is not set: implement fallback
            # logger.warning(f"{getattr(self, 'name', self.__class__.__name__)}: Contract failed. Returning default/fallback.")
            return MyOutput(result="Error: Processing failed due to contract violation.", is_valid=False)

        # Contract succeeded, self.contract_result holds the validated output
        # You can do further processing on self.contract_result if needed,
        # or simply return it.
        final_result: MyOutput = self.contract_result
        final_result.result += " (Forward processed)" # Example of further work
        return final_result
```

## Contract Execution Flow

When you call an instance of your contracted class (e.g., `my_instance(input=my_input_data)`), the `wrapped_forward` method (created by the `@contract` decorator) executes the following sequence:

1.  **Initial Input Validation (`_is_valid_input`)**:
    *   Checks if the provided `input` kwarg is an instance of `LLMDataModel`. Fails fast if not.
    *   Extracts the `original_input` object.

2.  **Pre-condition Validation (`_validate_input`)**:
    *   The `current_input` (initially `original_input`) is passed to your `pre(input)` method.
    *   If `pre()` raises an exception and `pre_remedy=True`, `SemanticValidationFunction` attempts to correct the `current_input` based on the exception message from `pre()` and your class's `prompt`.
    *   If `pre()` raises and `pre_remedy=False` (or remedy fails), an `Exception("Pre-condition validation failed!")` is raised (this exception is then handled by `wrapped_forward`'s main `try...except` block).

3.  **Intermediate Action (`_act`)**:
    *   If your class defines an `act` method:
        *   Its signature is validated (parameter named `input`, `LLMDataModel` type hints for input and output).
        *   `act(current_input, **act_kwargs)` is called. `current_input` here is the output from the pre-condition validation step.
        *   The result of `act` becomes the new `current_input`.
        *   The actual type of `act`'s return value is checked against its annotation.
    *   If no `act` method, `current_input` remains unchanged.

4.  **Output Validation & Generation (`_validate_output`)**:
    *   This is a critical step, especially when `post_remedy=True`.
    *   It uses `TypeValidationFunction` and (if `post_remedy=True`) `SemanticValidationFunction`.
    *   The goal is to produce an object that matches your `forward` method's return type annotation (e.g., `MyOutput`).
    *   The `current_input` (which is the output from `_act`, or from `_validate_input` if no `act`) and your class's `prompt` are used to guide an LLM call to generate/validate data conforming to the target output type.
    *   Your `post(output)` method is called with the LLM-generated/validated output object.
    *   If `post()` raises an exception and `post_remedy=True`, remediation is attempted.
    *   If all these steps (type validation, LLM generation, `post` validation, remedies) succeed:
        *   `self.contract_successful` is set to `True`.
        *   `self.contract_result` is set to the final, validated output object.
        *   This output is typically assigned to `final_output` within the `try` block of `wrapped_forward` (the method created by the decorator).

5.  **Exception Handling in Main Path (`wrapped_forward`'s `try...except`)**:
    *   Steps 2, 3, and 4 (pre-validation, act, and output validation/generation) are wrapped in a `try...except Exception as e:` block within the decorator's logic.
    *   If any exception occurs during these steps (e.g., an unrecoverable failure in `_validate_input`, `_act`, or `_validate_output`), the `logger.error` records it, and `self.contract_successful` is set to `False`.

6.  **Final Execution (`finally` block of `wrapped_forward`)**:
    *   This block **always executes**, regardless of success or failure in the preceding `try` block.
    *   It determines the `forward_input` for your original `forward` method:
        *   If `self.contract_successful` is `True`, `forward_input` is the `current_input` that successfully passed through `_act` and was used by `_validate_output`.
        *   If `self.contract_successful` is `False`, `forward_input` is the `original_input`.
    *   Your class's original `forward(self, input=forward_input, **kwargs)` method is called.
    *   The value returned by *your* `forward` method becomes the ultimate return value of the contract call.
    *   A final output type check is performed on this returned value against your `forward` method's declared return type annotation.

## Practical Example: Sentiment Analyzer

```python
import logging
from typing import Optional, List
from pydantic import Field

from symai import Expression
# from symai.components import MetadataTracker # If you want to track LLM usage
from symai.models import LLMDataModel
from symai.strategy import contract

# Configure basic logging for visibility if running standalone
# In a larger application, logging might be configured globally.
# import logging # Make sure logging is imported at the top
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__) # Use module-level logger for examples

class SentimentInput(LLMDataModel):
    text: str = Field(description="The text to analyze for sentiment.")

class SentimentAnalysis(LLMDataModel):
    sentiment: str = Field(description="The detected sentiment: 'positive', 'negative', or 'neutral'.")
    confidence: Optional[float] = Field(default=None, description="Confidence score of the sentiment analysis [0.0, 1.0].")
    keywords: List[str] = Field(default_factory=list, description="Keywords contributing to the sentiment.")

@contract(post_remedy=True, verbose=True) # verbose=True helps in debugging
class SentimentAnalyzer(Expression):
    def __init__(self, min_confidence: float = 0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_confidence = min_confidence
        self.processed_count = 0
        # It's good practice to use the symai logger or a configured application logger.
        # For example, at the module level:
        # from symai.utils.logger import logger
        # Or pass a logger instance if your application uses one.

    @property
    def prompt(self) -> str:
        return "Analyze the sentiment of the provided text. Identify if it's positive, negative, or neutral. Extract keywords that support this sentiment and provide a confidence score."

    def pre(self, input: SentimentInput) -> bool:
        if not input.text.strip():
            raise ValueError("Input text for sentiment analysis cannot be empty or just whitespace.")
        if len(input.text) < 3: # Arbitrary short text check
            raise ValueError("Input text is too short for meaningful sentiment analysis (min 3 chars).")
        return True

    # No 'act' method in this simple example

    def post(self, output: SentimentAnalysis) -> bool:
        valid_sentiments = ["positive", "negative", "neutral"]
        if output.sentiment not in valid_sentiments:
            raise ValueError(f"Sentiment must be one of {valid_sentiments}, got '{output.sentiment}'. Please correct the sentiment.")
        if output.confidence is not None:
            if not (0.0 <= output.confidence <= 1.0):
                raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {output.confidence}. Please provide a valid score.")
            if output.confidence < self.min_confidence:
                raise ValueError(f"Sentiment confidence {output.confidence} is below the minimum threshold of {self.min_confidence}. Please reassess confidence or ensure higher certainty for the chosen sentiment.")
        if output.sentiment != "neutral" and not output.keywords:
            raise ValueError("Keywords must be provided for non-neutral sentiments. Please extract relevant keywords.")
        return True

    def forward(self, input: SentimentInput, **kwargs) -> SentimentAnalysis:
        self.processed_count += 1 # Track calls to the original forward
        # Example of using a logger (ensure logger is defined, e.g., self.logger or module-level logger)
        # logger.info(f"Original forward in SentimentAnalyzer called for: '{input.text[:30]}...'")

        if not self.contract_successful or self.contract_result is None:
            # logger.warning("SentimentAnalyzer: Contract failed. Returning default/fallback sentiment.")
            return SentimentAnalysis(sentiment="neutral", confidence=0.0, keywords=["unknown_due_to_error"])

        # Contract was successful, self.contract_result is a validated SentimentAnalysis object
        # logger.info(f"SentimentAnalyzer: Contract successful. Result: {self.contract_result.sentiment} (Confidence: {self.contract_result.confidence})")
        return self.contract_result

# Example Usage (outside the class definition):
if __name__ == '__main__':
    # Configure logging for the example run (place at the top of your script)
    import logging # Ensure logging is imported
    from symai.components import MetadataTracker # For tracking LLM usage

    logging.basicConfig(level=logging.INFO)
    # Enable detailed logs from the contract strategy itself if needed
    symai_strategy_logger = logging.getLogger('symai.strategy')
    symai_strategy_logger.setLevel(logging.INFO) # or logging.DEBUG for more detail

    analyzer = SentimentAnalyzer(min_confidence=0.6)

    print("\n--- Testing with good input ---")
    good_text_input = SentimentInput(text="SymbolicAI is an amazing and powerful framework that simplifies complex AI tasks!")
    with MetadataTracker() as tracker:
        result = analyzer(input=good_text_input)
    print(f"Final Sentiment: {result.sentiment}, Confidence: {result.confidence}, Keywords: {result.keywords}")
    print(f"LLM Usage: {tracker.usage}")
    analyzer.contract_perf_stats() # Display performance for this call

    print("\n--- Testing with potentially problematic input (post_remedy=True will attempt to fix) ---")
    # This text might get a low confidence or unexpected sentiment from a generic LLM.
    # The post-condition for confidence or keywords might trigger remedies.
    problem_text_input = SentimentInput(text="The event was... an event.")
    with MetadataTracker() as tracker:
        result_problem = analyzer(input=problem_text_input)
    print(f"Final Sentiment (problematic): {result_problem.sentiment}, Confidence: {result_problem.confidence}, Keywords: {result_problem.keywords}")
    print(f"LLM Usage: {tracker.usage}")
    analyzer.contract_perf_stats()

    print("\n--- Testing with input designed to fail pre-condition ---")
    # analyzer is instantiated with @contract(post_remedy=True), default pre_remedy=False.
    # An error from pre() will be caught by wrapped_forward's main try-except block (due to strategy.py changes),
    # contract_successful will be False, and the fallback in SentimentAnalyzer.forward will be used.
    bad_text_input = SentimentInput(text="!")
    result_bad = analyzer(input=bad_text_input)
    print(f"Final Sentiment (bad pre): {result_bad.sentiment} (Keywords: {result_bad.keywords})")
    # We can check if the contract reported failure:
    print(f"Contract success for bad_text_input: {analyzer.contract_successful}") # Should be False
    analyzer.contract_perf_stats()
```

This example illustrates defining input/output models, `pre`/`post` conditions, a `prompt`, and the crucial logic within `forward` to handle both successful contract execution and failures. The `@contract` decorator, combined with these components, provides a robust framework for building reliable LLM-powered applications.
