# Operations

---

## ⚠️  Outdated or Deprecated Documentation ⚠️
This documentation is outdated and may not reflect the current state of the SymbolicAI library. This page might be revived or deleted entirely as we continue our development. We recommend using more modern tools that infer the documentation from the code itself, such as [DeepWiki](https://deepwiki.com/ExtensityAI/symbolicai). This will ensure you have the most accurate and up-to-date information and give you a better picture of the current state of the library.

---

## Overview
Operations form the core of our framework and serve as the building blocks of our API. These operations define the behavior of symbols by acting as contextualized functions that accept a `Symbol` object and send it to the neuro-symbolic engine for evaluation. Operations then return one or multiple new objects, which primarily consist of new symbols but may include other types as well. Polymorphism plays a crucial role in operations, allowing them to be applied to various data types such as strings, integers, floats, and lists, with different behaviors based on the object instance.

Operations are executed using the `Symbol` object's `value` attribute, which contains the original data type converted into a string representation and sent to the engine for processing. As a result, all values are represented as strings, requiring custom objects to define a suitable `__str__` method for conversion while preserving the object's semantics.

Inheritance is another essential aspect of our API, which is built on the `Symbol` class as its base. All operations are inherited from this class, offering an easy way to add custom operations by subclassing `Symbol` while maintaining access to basic operations without complicated syntax or redundant functionality. Subclassing the `Symbol` class allows for the creation of contextualized operations with unique constraints and prompt designs by simply overriding the relevant methods. However, it is recommended to subclass the `Expression` class for additional functionality.

Defining custom operations can be done through overriding existing Python methods and providing a custom prompt object with example code. Here is an example of creating a custom `==` operation by overriding the `__eq__` method:

```python
class Demo(ai.Symbol):
    def __eq__(self, other) -> bool:
        @ai.equals(examples=ai.Prompt([
              "1 == 'ONE' =>True",
              "'six' == 7 =>False",
              "'Acht' == 'eight' =>True",
              ...
          ])
        )
        def _func(_, other) -> bool:
            return False # default behavior on failure
        return _func(self, other)
```

## Basic Implementation

Basic operations in `Symbol` are implemented by defining local functions and decorating them with corresponding operation decorators from the `symai/core.py` file, a collection of predefined operation decorators that can be applied rapidly to any function. Using local functions instead of decorating main methods directly avoids unnecessary communication with the neural engine and allows for default behavior implementation. It also helps cast operation return types to symbols or derived classes, using the `self.sym_return_type(...)` method for contextualized behavior based on the determined return type. More details can be found in the [`Symbol` class](https://github.com/ExtensityAI/symbolicai/blob/main/symai/symbol.py).

The following section demonstrates that most operations in `symai/core.py` are derived from the more general `few_shot` decorator.

## Operation Types

### {#custom-operations} Custom Operations

Defining custom operations is also possible, such as creating an operation to generate a random integer between 0 and 10:

```python
class Demo(ai.Expression):
    @ai.zero_shot(prompt="Generate a random integer between 0 and 10.",
                  constraints=[
                      lambda x: x >= 0,
                      lambda x: x <= 10
                  ])
    def get_random_int(self) -> int:
        pass

demo = Demo()
random_int = demo.get_random_int()
print(random_int)  # Output: A random integer between 0 and 10
```

The Symbolic API employs Python `Decorators` to define operations, utilizing the `@ai.zero_shot` decorator to create custom operations that do not require demonstration examples when the prompt is self-explanatory. In this example, the `zero_shot` decorator accepts two arguments: `prompt` and `constraints`. The former defines the prompt dictating the desired operation behavior, while the latter establishes validation constraints for the computed outcome, ensuring it meets expectations.

If a constraint is not satisfied, the implementation will utilize the specified `default` fallback or default value. If neither is provided, the Symbolic API will raise a `ConstraintViolationException`. The return type is set to `int` in this example, so the value from the wrapped function will be of type int. The implementation uses auto-casting to a user-specified return data type, and if casting fails, the Symbolic API will raise a `ValueError`. If no return type is specified, the return type defaults to `Any`.

### Few-Shot Operations

The `@ai.few_shot` decorator is a generalized version of the `@ai.zero_shot` decorator, used to define custom operations that require demonstration examples. To provide a clearer understanding, we present the function signature of the `few_shot` decorator:

```python
def few_shot(prompt: str,
             examples: Prompt,
             constraints: List[Callable] = [],
             default: Optional[object] = None,
             limit: int = 1,
             pre_processors: Optional[List[PreProcessor]] = None,
             post_processors: Optional[List[PostProcessor]] = None,
             **decorator_kwargs):
```

The `prompt` and `constraints` attributes behave similarly to those in the `zero_shot` decorator. The `examples` and `limit` arguments are new. The `examples` argument defines a list of demonstrations used to condition the neural computation engine, while the `limit` argument specifies the maximum number of examples returned, given that there are more results. The `pre_processors` argument accepts a list of `PreProcessor` objects for pre-processing input before it's fed into the neural computation engine. The `post_processors` argument accepts a list of `PostProcessor` objects for post-processing output before returning it to the user. Lastly, the `decorator_kwargs` argument passes additional arguments from the decorator kwargs, which are streamlined towards the neural computation engine and other engines.

To provide a more comprehensive understanding of our conceptual implementation, refer to the flow diagram below, containing the most important classes:

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img9.png" width="600px">

The colors indicate logical groups of data processing steps. `Yellow` represents input and output data, `blue` shows places where one can customize or prepare the input of the engine, `green` indicates post-processing steps of the engine response, `red` displays the application of constraints (including attempted casting of the `return type signature` if specified in the decorated method), and `grey` denotes the custom method defining all properties, thus having access to all the previously mentioned objects.

To conclude this section, here is an example of how to write a custom Japanese name generator using our `@ai.zero_shot` decorator:

```python
import symai as ai
class Demo(ai.Symbol):
    @ai.few_shot(prompt="Generate Japanese names: ",
                 examples=ai.Prompt(
                   ["愛子", "和花", "一郎", "和枝"]
                 ),
                 limit=2,
                 constraints=[lambda x: len(x) > 1])
    def generate_japanese_names(self) -> list:
        return ['愛子', '和花'] # dummy implementation
```

If the neural computation engine cannot compute the desired outcome, it will revert to the `default` implementation or default value. If no default implementation or value is found, the method call will raise an exception.

## Prompt Design

The `Prompt` class is used to perform all the above operations. Acting as a container for information required to define a specific operation, the `Prompt` class also serves as the base class for all other Prompt classes.

Here's an example of defining a `Prompt` to enforce the neural computation engine to compare two values:

```python
class CompareValues(ai.Prompt):
    def __init__(self) -> ai.Prompt:
        super().__init__([
            "4 > 88 =>False",
            "-inf < 0 =>True",
            "inf > 0 =>True",
            "4 > 3 =>True",
            "1 < 'four' =>True",
            ...
        ])
```

When calling the `<=` operation on two Symbols, the neural computation engine evaluates the symbols in the context of the `CompareValues` prompt.

```python
res = ai.Symbol(1) <= ai.Symbol('one')
```

This statement evaluates to `True` since the fuzzy compare operation conditions the engine to compare the two Symbols based on their semantic meaning.

```bash
:[Output]:
True
```

In general, the semantics of Symbol operations may vary depending on the context hierarchy of the expression class and the operations used. To better illustrate this, we display our conceptual prompt design in the following figure:

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img4.png" width="350px">

The figure illustrates the hierarchical prompt design as a container for information provided to the neural computation engine to define a task-specific operation. The `yellow` and `green` highlighted boxes indicate mandatory string placements, dashed boxes represent optional placeholders, and the `red` box marks the starting point of model prediction.

Three main prompt designs are considered: `Context-based Prompts`, `Operational Prompts`, and `Templates`. Prompts can be curated either by inheritance or composition. For example, `Static Context` can be defined by inheriting the `Expression` class and overriding the `static_context` property. An `Operation` and `Template` prompt can be created by providing a `PreProcessor` to modify input data.

Each prompt concept is explained in more detail below:

- `Context-based Prompts (Static, Dynamic, and Payload)` are considered optional and can be defined either statically (by subclassing the Expression class and overriding the `static_context` property) or at runtime (by updating the `dynamic_context` property or passing `payload` kwargs to a method). As an example of using the `payload` kwargs via method signature:

  ```python
  # creating a query to ask if an issue was resolved or not
  sym = Symbol("<some-community-conversation>")
  q = sym.query("Was the issue resolved?")
  # write manual condition to check if the issue was resolved
  if 'not resolved' in q:
      # do a new query but payload the previous query answer to the new query
      sym.query("What was the resolution?", payload=q)
      ...
  else:
      pass # all good
  ```

  Regardless of how the context is set, the contextualized prompt defines the desired behavior of Expression operations. For example, one can operate within a domain-specific language context without having to override each base class method. See more details [here](https://extensityai.gitbook.io/symbolicai/tutorials/data_query).

- `Operation` prompts define the behavior of atomic operations and are mandatory to express the nature of such operations. For example, the `+` operation is used to add two Symbols together, so its prompt explains this behavior. `Examples` provide an optional structure giving the neural computation engine a set of demonstrations used to condition it properly. For instance, the `+` operation prompt can be conditioned on adding numbers by providing demonstrations like `1 + 1 = 2`, `2 + 2 = 4`, etc.

- `Template` prompts are optional and encapsulate the resulting prediction to enforce a specific format. For example, to generate HTML tags, one can use the curated `<html>{{placeholder}}</html>` template. This template ensures that the neural computation engine starts the generation process within the context of an HTML tag format, avoiding the production of irrelevant descriptions regarding its task.
