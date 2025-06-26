# Primitives Cheat-Sheet

> _Primitives are the atoms of the SymbolicAI API – small, orthogonal operations that can be freely combined to express very rich behaviour.
Every `Symbol` automatically inherits them, so there is no import-gymnastics required: just create a symbol and start hacking._

This page gives you a **hands-on overview** of the most useful primitives that ship with
`symai`. All examples are *copy-pastable* – open a Python REPL and play with them!  The table of
contents below mirrors the internal organisation of `symai/ops/primitives.py`, so if you ever need
more detail you know where to look.

> ❗️**NOTE**❗️We will focus mostly on the **semantic** behaviour of the primitives, but you can also use them in **syntactic** mode, which is the default.
Syntactic mode behaves like a normal Python value, unless otherwise specified.
Be sure to check the examples below to see how the primitives behave.

<!-- start primitives overview table -->
## Supported Primitives - SymbolicAI
This table summarizes all supported primitives, grouped by category. Use Python operators in
either syntactic (`Symbol(...).syn` or `Symbol(...)`) or semantic (`Symbol(...).sem` or `Symbol(..., semantic=True)`) mode. Semantic mode invokes neuro-symbolic reasoning.
**Checkmark** columns show which mode is supported; description explains the behavior.

| Primitive/Operator | Category         | Syntactic | Semantic | Description |
|--------------------|-----------------|:---------:|:--------:|-------------|
| `.sem` / `.syn`    | Casting         | ✓         | ✓        | Switches a symbol between syntactic (literal) and semantic (neuro-symbolic) behavior. |
| `==`               | Comparison      | ✓         | ✓        | Tests for equality. Syntactic: literal match. Semantic: fuzzy/conceptual equivalence (e.g. 'Hi' == 'Hello'). |
| `!=`               | Comparison      | ✓         | ✓        | Tests for inequality. Syntactic: literal not equal. Semantic: non-equivalence or opposite concepts. |
| `>`                | Comparison      | ✓         | ✓        | Greater-than. Syntactic: numeric/string compare. Semantic: abstract comparison (e.g. 'hot' > 'warm'). |
| `<`                | Comparison      | ✓         | ✓        | Less-than. Syntactic: numeric/string compare. Semantic: abstract ordering (e.g. 'cat' < 'dog'). |
| `>=`               | Comparison      | ✓         | ✓        | Greater or equal. Syntactic or conceptual. |
| `<=`               | Comparison      | ✓         | ✓        | Less or equal. Syntactic or conceptual. |
| `in`               | Membership      | ✓         | ✓        | Syntactic: element in list/string. Semantic: membership by meaning (e.g. 'fruit' in ['apple', ...]). |
| `~`                | Invert   | ✓         | ✓        | Negation: Syntactic: logical NOT/bitwise invert. Semantic: conceptual inversion (e.g. 'True' ➔ 'False', 'I am happy.' ➔ 'Happiness is me.'). |
| `+`                | Arithmetic      | ✓         | ✓        | Syntactic: numeric/string/list addition. Semantic: meaningful composition, blending, or conceptual merge. |
| `-`                | Arithmetic      | ✓         | ✓        | Syntactic: subtraction/negate. Semantic: replacement or conceptual opposition. |
| `*`                | Arithmetic      | ✓         |         | Syntactic: multiplication/repeat. Semantic: expand or strengthen meaning. |
| `@`                | Arithmetic      | ✓         |         | Sytactic: string concatenation. |
| `/`                | Arithmetic      | ✓         |         | Syntactic: division. On strings, it splits the string based on delimiter (e.g. `Symbol('a b') / ' '` -> `['a', 'b']`))). |
| `//`               | Arithmetic      | ✓         |         | Floor division. |
| `%`                | Arithmetic      | ✓         |         | Modulo. Semantic: find remainder or part, can be used creatively over concepts. |
| `**`               | Arithmetic      | ✓         |         | Power operation. Semantic: (hypernym or intensifier, depending on domain). |
| `&`                | Logical/Bitwise | ✓         | ✓        | Syntactic: bitwise/logical AND. Semantic: logical conjunction, inference, e.g., context merge. |
| `\|`                | Logical/Bitwise | ✓         | ✓        | Syntactic: bitwise/logical OR. Semantic: conceptual alternative/option. |
| `^`                | Logical/Bitwise | ✓         | ✓        | Syntactic: bitwise XOR. Semantic: exclusive option/concept distinction. |
| `<<`               | Shift           | ✓         | ✓        | Syntactic: left-shift (integers). Semantic: prepend or rearrange meaning/order. |
| `>>`               | Shift           | ✓         | ✓        | Syntactic: right-shift (integers). Semantic: append or rearrange meaning/order. |
| `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=` | In-place Arithmetic | ✓ | ✓ | In-place enhanced assignment, some syntactic and some semantic (see above). |
| `&=`, `\|=`, `^=`   | In-place Logical| ✓         | ✓        | In-place enhanced bitwise/logical assignment, both syntactic and semantic. |
| `.cast(type)`, `.to(type)` | Casting         | ✓         |         | Cast to a specified type (e.g., int, float, str). |
| `.str()`, `.int()`, `.float()`, `.bool()` | Casting    |     ✓      |         | Cast to basic types. |
| `.ast()`             | Casting           | ✓         |         | Parse a Python literal string into a native object. |
| `symbol[index]`, `symbol[start:stop]` | Iteration   | ✓         | ✓        | Get item or slice (list, tuple, dict, numpy array). |
| `symbol[index] = value` | Iteration        | ✓         | ✓        | Set item or slice. |
| `del symbol[index]`      | Iteration        | ✓         | ✓        | Delete item or key. |
| `.split(delimiter)`      | String Helper    | ✓         |         | Split a string or sequence into a list. |
| `.join(delimiter)`      | String Helper    | ✓         |         | Join a list of strings into a string. |
| `.startswith(prefix)`    | String Helper    | ✓         | ✓        | Check if a string starts with given prefix (in both modes). |
| `.endswith(suffix)`      | String Helper    | ✓         | ✓        | Check if a string ends with given suffix (in both modes). |
| `.equals(string, context?)` | Comparison     |           | ✓        | Semantic/contextual equality beyond `==`. |
| `.contains(element)`         | Comparison      |           | ✓        | Semantic contains beyond `in`. |
| `.isinstanceof(query)` | Comparison     |           | ✓        | Semantic type checking. |
| `.interpret(prompt, accumulate?)`| Expression Handling |        | ✓ | Interpret prompts/expressions|
| `.get_results()`        | Expression Handling |     ✓     |       | Retrieve accumulated interpretation results. |
| `.clear_results()`        | Expression Handling |    ✓      |        | Clear accumulated interpretation results. |
| `.clean()`                | Data Handling   |           | ✓        | Clean text (remove extra whitespace, newlines, tabs). |
| `.summarize(context?)`    | Data Handling   |           | ✓        | Summarize text (optionally with context). |
| `.outline()`              | Data Handling   |           | ✓        | Generate outline from structured text. |
| `.filter(criteria, include?)` | Data Handling |   | ✓        | Filter text by criteria (exclude/include). |
| `.map(instruction, prompt?)`                  | Data Handling    |         | ✓        | Semantic mapping over iterables. |
| `.modify(changes)`        | Data Handling   |           | ✓        | Apply modifications according to prompt. |
| `.replace(old, new)`      | Data Handling   |          | ✓        | Replace substrings in data. |
| `.remove(information)`           | Data Handling   |           | ✓        | Remove specified text. |
| `.include(information)`          | Data Handling   |           | ✓        | Include additional information. |
| `.combine(information)`          | Data Handling   |           | ✓        | Combine with another text fragment. |
| `.unique(keys?)`          | Uniqueness      |           | ✓        | Extract unique elements or entries. |
| `.compose()`              | Uniqueness      |           | ✓        | Compose a coherent narrative. |
| `.rank(measure?, order?)`   | Pattern Matching|           | ✓        | Rank items by a given measure/order. |
| `.extract(pattern)`       | Pattern Matching|           | ✓        | Extract info matching a pattern. |
| `.correct(context, exception)` | Pattern Matching |      | ✓        | Correct code/text based on prompt/exception. |
| `.translate(language)` | Pattern Matching |         | ✓ | Translate text into another language. |
| `.choice(cases, default)` | Pattern Matching|           | ✓        | Select best match from provided cases. |
| `.query(context, prompt?, examples?)`  | Query Handling  |           | ✓        | Query structured data with a question or prompt. |
| `.convert(format)`        | Query Handling  |           | ✓        | Convert data to specified format (YAML, XML, etc.). |
| `.transcribe(modify)` | Query Handling |          | ✓        | Transcribe/reword text per instructions. |
| `.analyze(exception, query?)` | Execution Control |      | ✓        | Analyze code execution and exceptions. |
| `.execute()`, `.fexecute()` | Execution Control |       | ✓        | Execute code (with fallback). |
| `.simulate()`             | Execution Control |         | ✓        | Simulate code or process semantically. |
| `.sufficient(query)`   | Execution Control |         | ✓        | Check if information is sufficient. |
| `.list(condition)`         | Execution Control |         | ✓        | List items matching criteria. |
| `.foreach(condition, apply)`| Execution Control |         | ✓        | Apply action to each element. |
| `.stream(expr, token_ratio)` | Execution Control |      | ✓        | Stream-process large inputs. |
| `.ftry(expr, retries)`    | Execution Control |         | ✓        | Fault-tolerant execution with retries. |
| `.dict(context, **kwargs)` | Dict Handling    |         | ✓        | Convert text/list into a dict semantically. |
| `.template(template, placeholder?)` | Template Styling |    ✓  | | Fill in placeholders in a template string. |
| `.style(description, libraries?)` | Template Styling |        | ✓ | Style text/code (e.g., syntax highlighting). |
| `.cluster(**clustering_kwargs?)`              | Data Clustering  |         | ✓        | Cluster data into groups semantically. (uses sklearn's DBSCAN)|
| `.embed()`                | Embedding        |         | ✓        | Generate embeddings for text/data. |
| `.embedding`              | Embedding        |         | ✓        | Retrieve embeddings as a numpy array. |
| `.similarity(other, metric?, normalize?)` | Embedding    |         | ✓        | Compute similarity between embeddings. |
| `.distance(other, kernel?)`  | Embedding     |         | ✓        | Compute distance between embeddings. |
| `.zip()`                  | Embedding        |         | ✓        | Package id, embedding, query into tuples. |
| `.open(path?)`             | IO Handling      | ✓       |         | Open a file and read its contents. |
| `.input(message?)`                | IO Handling      | ✓       |         | Read user input interactively. |
| `.save(path, serialize?, replace?)` | Persistence | ✓ |  | Save a symbol to file (pickle/text). |
| `.load(path)`             | Persistence      | ✓       |         | Load a symbol from file. |
| `.expand()`               | Persistence      |         | ✓        | Generate and attach code based on prompt. |
| `.output()` | Output Handling | ✓    | | Handle/capture output with handler. |

---
## 1. Syntactic **vs.** Semantic Symbols

`Symbol` comes in **two flavours**:

1. **Syntactic** – behaves like a normal Python value (string, list, int ‐ whatever you passed in).
2. **Semantic**  – is wired to the neuro-symbolic engine and therefore *understands* meaning and
   context.

Why is syntactic the default?  Because Python operators (`==`, `~`, `&`, …) are overloaded in
`symai`.  If we would immediately fire the engine for *every* bitshift or comparison, code would be
slow and could produce surprising side-effects.  Starting syntactic keeps things safe and fast; you
opt-in to semantics only where you need them.

### How to switch to the semantic view

1. **At creation time**

   ```python
   from symai import Symbol

   s = Symbol("Cats are adorable", semantic=True) # already semantic
   ```

2. **On demand with the `.sem` projection** – the twin `.syn` flips you back:

   ```python
   s = Symbol("Cats are adorable") # default = syntactic
   print(s.startswith("Cats")) # syntactic => True
   print(s.sem.startswith("animal")) # semantic => True
   print("feline" in s.sem) # semantic => True
   result = (s / " ")[0] # syntactic operator split, syntactic index
   print(result) # => 'Cats'
   ```

Because the projections return the *same underlying object* with just a different behavioural
coat, you can weave complex chains of syntactic and semantic operations on a single symbol.

## 2. Inversion and Negation

#### Inversion (`~`)

```python
s = Symbol("I am standing on the shoulders of giants.", semantic=True)
print(~s) # => Giants are standing on my shoulders.
```

#### Negation (`-`)

```python
s = Symbol("I am happy.", semantic=True)
print(-s) # => I am not happy.
```

## 3. Comparison Operations

### Equality (`==`) and Inequality (`!=`)

The equality and inequality operators showcase one of the most powerful aspects of semantic symbols: **conceptual equivalence** rather than just literal matching.

#### Semantic Equality - Conceptual Equivalence

```python
from symai import Symbol

# Semantic greeting comparison - understands meaning
greeting_sem = Symbol('Hello there!', semantic=True)
greeting_variant = 'Hi there!'

print(greeting_sem == greeting_variant) # => True

# Semantic farewell comparison
farewell_sem = Symbol('Goodbye friend!', semantic=True)
farewell_variant = 'See you later!'

print(farewell_sem == farewell_variant) # => True

# Works with lists too - semantic understanding
list_sem = Symbol([1, 2, 3], semantic=True)
list_different = Symbol([3, 2, 1])

print(list_sem == list_different) # => False
```

#### Semantic Inequality (`!=`)

The inequality operator works as the logical inverse of equality with conceptual understanding:

```python
# Semantic inequality - conceptual differences
greeting_sem = Symbol('Hello there!', semantic=True)
greeting_variant = 'Hi there!'

print(greeting_sem != greeting_variant) # => False

farewell_sem = Symbol('Goodbye friend!', semantic=True)
farewell_variant = 'See you later!'

print(farewell_sem != farewell_variant) # => False
```

#### Advanced Contextual Equality - Custom Context Comparison

Beyond basic semantic equality, symbols support **contextual comparison** where you can specify the context for evaluation:

```python
from symai import Symbol

# Contextual greeting comparison
greeting = Symbol('Hello, good morning!')
similar_greeting = 'Hi there, good day!'

# Compare with specific greeting context
result = greeting.equals(similar_greeting, context='greeting context')
print(result) # => True

# Compare with different contexts for nuanced evaluation
formal_greeting = Symbol('Good morning, sir.')
casual_greeting = 'Hey, what\'s up?'

# Context-aware politeness comparison
politeness_comparison = formal_greeting.equals(casual_greeting, context='politeness level')
print(politeness_comparison) # => False
```

## 4. Membership Operations

### The `in` Operator - Semantic Containment

The `in` operator demonstrates how semantic symbols can understand **conceptual membership** beyond literal substring or element matching.

#### Semantic Membership - Conceptual Containment

```python
from symai import Symbol

# Semantic string containment - understands concepts
str_sem = Symbol('apple banana cherry', semantic=True)
concept_search = 'fruit'

print(concept_search in str_sem) # => True (understands apple/banana are fruits)

# Semantic list containment - conceptual understanding
list_sem = Symbol(['apple', 'banana', 'cherry'], semantic=True)

print(concept_search in list_sem) # => True (conceptual membership)

# More examples of semantic containment
animals_sem = Symbol('cat dog bird fish', semantic=True)
print('pet' in animals_sem)     # => True (cats and dogs are pets)
print('mammal' in animals_sem)  # => True (cat and dog are mammals)
print('vehicle' in animals_sem) # => False (no vehicles in the list)
```

#### Enhanced Semantic Containment - Multi-Type Support

The `contains()` method provides advanced semantic containment that works across different data types:

```python
from symai import Symbol

# Semantic containment in text with concept understanding
text = Symbol('The vehicle moved quickly down the road')
print(text.contains('car')) # => True (vehicle can semantically contain car)

# Containment with mixed data types
mixed_data = Symbol([1, 'two', 3.0, True])
print(mixed_data.contains('two')) # => True (exact string match)
print(mixed_data.contains(True)) # => True (exact boolean match)
```

## 5. Ordering and Comparison Operations

### Greater Than (`>`) and Less Than (`<`) and Greater or Equal (`>=`) and Less or Equal (`<=`)

Semantic symbols can understand abstract ordering and magnitude relationships beyond numeric comparisons.

```python
from symai import Symbol

# Semantic size comparison - understands relative magnitude
large_sem = Symbol('enormous', semantic=True)
small_sem = Symbol('tiny', semantic=True)

print(large_sem > small_sem) # => True (enormous is conceptually larger than tiny)

# Semantic animal size comparison
cat_sem = Symbol('cat', semantic=True)
dog_sem = Symbol('dog', semantic=True)

print(cat_sem < dog_sem) # => True (dogs are generally larger than cats)

# Temperature comparisons - semantic understanding of hot/cold
cold_sem = Symbol('freezing', semantic=True)
hot_sem = Symbol('scorching', semantic=True)

print(cold_sem < hot_sem) # => True (freezing is less than scorching)

# Semantic number understanding
word_number = Symbol('fifty', semantic=True)
regular_number = 45

print(word_number > regular_number) # => True (understands "fifty" means 50)

# Speed comparisons with semantic understanding
fast_sem = Symbol('lightning speed', semantic=True)
slow_sem = Symbol('snail pace', semantic=True)

print(fast_sem >= slow_sem) # => True (lightning speed is much faster)

# Strength comparisons
weak_sem = Symbol('fragile', semantic=True)
strong_sem = Symbol('robust', semantic=True)

print(weak_sem <= strong_sem) # => True (fragile is weaker than robust)
```

## 6. Shift Operations - Semantic Inclusion

### Left Shift (`<<`) and Right Shift (`>>`)

Shift operations perform **semantic inclusion** by incorporating one symbol's content into another. The direction of the operator indicates where the inclusion occurs - left shift (`<<`) prepends content, while right shift (`>>`) appends content.

#### Semantic Inclusion Operations

```python
from symai import Symbol

# Left shift - prepend/include at the beginning
data_stream = Symbol('data stream', semantic=True)
left_direction = Symbol('left')
result = data_stream << left_direction
print(result) # => "left data stream"

# Right shift - append/include at the end
info_flow = Symbol('information flow', semantic=True)
right_direction = Symbol('right')
result = info_flow >> right_direction
print(result) # => "information flow right"

# Priority and urgency - prepend urgent to task
priority_task = Symbol('high priority task', semantic=True)
urgency = Symbol('urgent')
result = priority_task << urgency
print(result) # => "urgent high priority task"

# Process flow - append downstream to processing
data_process = Symbol('data processing', semantic=True)
downstream = Symbol('downstream')
result = data_process >> downstream
print(result) # => "data processing downstream"
```

#### Practical Use Cases

The shift operators provide a consistent way to indicate inclusion direction in high-level design:

- **Left Shift (`<<`)**: Use when you want to prepend, prioritize, or add context before
- **Right Shift (`>>`)**: Use when you want to append, extend, or add context after

This semantic approach allows for intuitive text manipulation and content combination while maintaining clear directional intent in your code design.

## 7. Bitwise and Logical Operations

Bitwise operators (`&`, `|`, `^`) work with both numeric values and logical statements, providing mathematical bitwise operations for integers and logical inference for semantic symbols.

### Bitwise AND (`&`) - Intersection and Logical Conjunction

```python
from symai import Symbol

# Semantic logical conjunction - combining facts and rules
horn_rule = Symbol('The horn only sounds on Sundays.', semantic=True)
observation = Symbol('I hear the horn.')
conclusion = horn_rule & observation # => Logical inference

# Combining evidence
evidence1 = Symbol('The suspect was seen at the scene.', semantic=True)
evidence2 = Symbol('His fingerprints were found.')
combined_evidence = evidence1 & evidence2 # => Combined evidence
```

### Bitwise OR (`|`) - Union and Logical Disjunction

```python
from symai import Symbol

# Semantic logical disjunction - alternative possibilities
option1 = Symbol('It might rain today.', semantic=True)
option2 = Symbol('It could be sunny.')
possibilities = option1 | option2 # => Multiple possibilities

# Alternative scenarios
scenario1 = Symbol('The meeting could be postponed.', semantic=True)
scenario2 = Symbol('The meeting might be canceled.')
alternatives = scenario1 | scenario2 # => Alternative outcomes
```

### Bitwise XOR (`^`) - Exclusive OR

```python
from symai import Symbol

# Semantic exclusive choice - either/or scenarios
exclusive1 = Symbol('Either it will rain today.', semantic=True)
exclusive2 = Symbol('Or it will be sunny.')
either_or = exclusive1 ^ exclusive2 # => Exclusive alternative

# Contradictory statements
statement1 = Symbol('The door is open.', semantic=True)
statement2 = Symbol('The door is closed.')
contradiction = statement1 ^ statement2 # => Mutually exclusive states
```

## 8. Arithmetic Operations

Basic arithmetic operations (`+`, `-`) provide both syntactic combination and semantic mathematical interpretation.

### Addition (`+`) and Subtraction (`-`)

```python
from symai import Symbol

enemy_text = Symbol('Hello my enemy', semantic=True)
result = enemy_text - 'enemy' + 'friend'
print(result) # => "Hello my friend"
```

### String Manipulation (`@`, `/`)

The following are syntactic sugar for string concatenation and splitting:

```python
from symai import Symbol

string = Symbol('Hello my ') @ 'friend' # Concatenation
split_string = string / ' ' # ['Hello', 'my', 'friend']
```
## 9. Indexing and Slicing Operations

The indexing and slicing operations (`__getitem__`, `__setitem__`, `__delitem__`) provide both syntactic data access and semantic key matching for dictionaries and lists.

### Getting Items (`symbol[key]`)
```python
from symai import Symbol

# Semantic dictionary access - finds conceptually related keys
sym_person = Symbol({'name': 'Alice', 'age': 30, 'city': 'NYC'}, semantic=True)
name_result = sym_person['Return any names'] # => 'Alice'
identity_result = sym_person['identity'] # => 'Alice' (matches 'name')
profession_result = sym_person['profession'] # => might return 'age' or relevant info

# Semantic list access - finds conceptually matching items
sym_animals = Symbol(['cat', 'dog', 'bird', 'fish'], semantic=True)
pet_result = sym_animals['domestic animal']

# Color matching example
sym_colors = Symbol({'red': '#FF0000', 'green': '#00FF00', 'blue': '#0000FF'}, semantic=True)
primary_color = sym_colors['primary color'] # => one of the hex values
```

### Setting Items (`symbol[key] = value`)
```python
from symai import Symbol

# Semantic dictionary modification - maps to conceptually similar keys
sym_weather = Symbol({'temperature': 20, 'humidity': 60, 'pressure': 1013}, semantic=True)
sym_weather['Change the temperature'] = 25 # => maps to 'temperature' key
```

### Deleting Items (`del symbol[key]`)
```python
from symai import Symbol

# Semantic dictionary deletion - removes conceptually matching keys
sym_person = Symbol({'first_name': 'John', 'last_name': 'Doe', 'age': 30}, semantic=True)
del sym_person['surname'] # => removes 'last_name' (conceptually equivalent)
# => {'first_name': 'John', 'age': 30}
```

## 10. Type Checking Operations

### Semantic Type Validation (`isinstanceof`)

The `isinstanceof()` method provides **semantic type checking** that goes beyond traditional Python type checking to understand conceptual types:

```python
from symai import Symbol

# Basic semantic type checking
number_sym = Symbol(42)
print(number_sym.isinstanceof('number')) # => True
print(number_sym.isinstanceof('string')) # => False

# Collection type checking
list_sym = Symbol(['apple', 'banana', 'cherry'])
print(list_sym.isinstanceof('list')) # => True

# Boolean/logical type checking
bool_sym = Symbol(True)
print(bool_sym.isinstanceof('logical value')) # => True

# Complex semantic type recognition
person_data = Symbol({'name': 'John', 'age': 30})
print(person_data.isinstanceof('person data')) # => True

# More natural language type queries
user_info = Symbol({'name': 'Alice', 'age': 25, 'city': 'Wonderland'})
print(user_info.isinstanceof('person')) # => True (understands structure represents a person)
```

## 11. Basic Symbolic Manipulations

The `interpret()` method provides a way to interpret and process symbolic expressions. Each step is atomic, meaning you can build complex interpretations by chaining them together.

```python
from symai import Symbol

# Symbolic reasoning and analogies
analogy = Symbol('gravity : Earth :: radiation : ?')
result = analogy.interpret()
print(result) # => Sun (or similar celestial body)

# Mathematical expression interpretation
math_expr = Symbol("∫(3x² + 2x - 5)dx")
solution = math_expr.interpret()
print(solution) # => x³ + x² - 5x + C
```

### Conditional Logic Processing

```python
# Conditional interpretation with different inputs
conditional = Symbol("If x < 0 then 'negative' else if x == 0 then 'zero' else 'positive'")

# Test with different values
print(conditional.interpret("x = -5")) # => negative
print(conditional.interpret("x = 0"))  # => zero
print(conditional.interpret("x = 10")) # => positive
```

### System Solving and Constraints

```python
# Mathematical system solving
system = Symbol("Find values for x and y where: x + y = 10, x - y = 4")
solution = system.interpret()
print(solution) # => x = 7, y = 3

# Logical reasoning with philosophical questions
reasoning = Symbol('If every event has a cause, and the universe began with an event, what philosophical question arises?')
philosophical_insight = reasoning.interpret()
print(philosophical_insight) # => What caused the first cause? (First Cause Argument)
```

The `interpret()` method supports can accumulate steps into one symbol:

```python
sym_accumulate = Symbol('Relativistic electron formula')
result1 = sym_accumulate.interpret(accumulate=True)
result2 = result1.interpret('Assume the momentum to be extremely large', accumulate=True)
result3 = result2.interpret('Expand the formula to account for both mass and momentum', accumulate=True) # Ideally, we should get back to the original formula
all_results = sym_accumulate.get_results() # get all accumulated results (3)
sym_accumulate.clear_results()
```

## 12. Data Processing and Manipulation Operations

### Text Cleaning and Normalization (`clean`)

The `clean()` method normalizes whitespace, removes extra spaces, and standardizes text formatting:

```python
sym_dirty = Symbol("This text has   multiple    spaces and\n\nextra newlines.\t\tAnd tabs.")
cleaned = sym_dirty.clean()
# => "This text has multiple spaces and extra newlines. And tabs."
```

### Content Summarization (`summarize`)

The `summarize()` method creates concise summaries of longer content, with optional context for focused summarization:

```python
sym_long = Symbol("Python is a high-level, interpreted programming language...")
summarized = sym_long.summarize()
# Creates a shorter version maintaining key information

# With context for focused summarization
context_summarized = sym_long.summarize(context="Focus on Python's use in data science")
# Emphasizes data science aspects in the summary
```

### Content Outlining (`outline`)

The `outline()` method extracts structure and key points from hierarchical content:

```python
sym_complex = Symbol("""
#Introduction to Machine Learning
Machine learning is a subset of artificial intelligence...
## Supervised Learning
### Classification
### Regression
## Unsupervised Learning
""")
outlined = sym_complex.outline()
# => ['- Machine Learning: subset of AI...', '- Supervised Learning: training...', ...]
```

### Content Filtering (`filter`)

The `filter()` method selectively includes or excludes content based on criteria:

```python
sym_mixed = Symbol("Dogs are loyal pets. Cats are independent pets. Hamsters are small pets.")

# Exclude content matching criteria (default behavior)
filtered_ex = sym_mixed.filter(criteria="Cats")
# => "Dogs are loyal pets. Hamsters are small pets."

# Include only content matching criteria
filtered_in = sym_mixed.filter(criteria="Dogs", include=True)
# => "Dogs are loyal pets."
```

### Content Modification (`modify`)

The `modify()` method applies specified changes to content:

```python
sym_original = Symbol("The quick brown fox jumps over the lazy dog.")
changes = "Change 'quick' to 'fast' and 'lazy' to 'sleeping'"
modified = sym_original.modify(changes=changes)
# => "The fast brown fox jumps over the sleeping dog."
```

### Text Replacement (`replace`)

The `replace()` method substitutes specific text with new content:

```python
sym_replace = Symbol("Python is a programming language. Python is easy to learn.")
replaced = sym_replace.replace("Python", "JavaScript")
# => "JavaScript is a programming language. JavaScript is easy to learn."
```

### Content Removal (`remove`)

The `remove()` method eliminates specified content:

```python
sym_extra = Symbol("This text contains [unnecessary information] that should be removed.")
removed = sym_extra.remove("[unnecessary information] ")
# => "This text contains that should be removed."
```

### Content Addition (`include`, `combine`)

Both methods add content, with subtle differences in behavior:

```python
sym_base = Symbol("This is the main content.")

# Include additional information
included = sym_base.include("This is additional information.")
# => "This is the main content. This is additional information."

# Combine with other content
combined = sym_base.combine("Second part of the content.")
# => "This is the main content. Second part of the content."
```

## 13. Pattern Matching and Intelligence Operations

### Content Ranking (`rank`)

The `rank()` method orders content based on specified measures and criteria:

```python
sym_numbers = Symbol("""
5. Learn Python basics
1. Install Python
3. Practice coding
2. Set up IDE
4. Join community
""")
ranked = sym_numbers.rank(measure='difficulty', order='desc')
# => Reorders by difficulty (descending)

# Ranking by priority
sym_tasks = Symbol("""
Important task: Complete project proposal
Urgent task: Fix critical bug
Optional task: Update documentation
Critical task: Deploy hotfix
""")
ranked_priority = sym_tasks.rank(measure='priority', order='asc')
```

### Pattern Extraction (`extract`)

The `extract()` method identifies and extracts specific patterns or information types:

```python
# Extract specific patterns like dates
sym_text = Symbol("""
Project deadline: 2024-03-15
Budget allocated: $50,000
Team size: 8 people
Status: In Progress
""")
extracted_dates = sym_text.extract("dates and deadlines")
# => "2024-03-15"
```

### Code Correction (`correct`)

The `correct()` method automatically fixes code errors based on exceptions:

```python
# Fix syntax errors
sym_code = Symbol("""
def calculate_sum(a b):
    return a + b
""")
corrected = sym_code.correct("Fix the code", exception=SyntaxError)
# => "def calculate_sum(a, b):\n    return a + b"

# Fix type errors
sym_type_error = Symbol("""
def process_data(items):
    return items.sort()
result = process_data([3, 1, 2])
print(result + 1) # TypeError: NoneType + int
""")
corrected_type = sym_type_error.correct("Fix the code", exception=TypeError)
# => Uses sorted() instead of sort() to return the sorted list
```

### Language Translation (`translate`)

The `translate()` method converts content between languages:

```python
sym_english = Symbol("Hello, how are you today?")
translated = sym_english.translate("Spanish")
# => "Hola, ¿cómo estás hoy?"
```

### Multi-Choice Classification (`choice`)

The `choice()` method classifies content into predefined categories:

```python
sym_weather = Symbol("Temperature: 85°F, Humidity: 70%, Conditions: Sunny")
cases = ["hot and humid", "mild", "cold and dry"]
weather_choice = sym_weather.choice(cases=cases, default="mild")
# => "hot and humid"

# Sentiment analysis
sym_sentiment = Symbol("This product exceeded all my expectations! Absolutely wonderful!")
sentiment_cases = ["positive", "neutral", "negative"]
sentiment_choice = sym_sentiment.choice(cases=sentiment_cases, default="neutral")
# => "positive"
```

## 12. Data Processing and Manipulation Operations

### Text Cleaning and Normalization (`clean`)

The `clean()` method normalizes whitespace, removes extra spaces, and standardizes text formatting:

```python
sym_dirty = Symbol("This text has   multiple    spaces and\n\nextra newlines.\t\tAnd tabs.")
cleaned = sym_dirty.clean()
# => "This text has multiple spaces and extra newlines. And tabs."
```

### Content Summarization (`summarize`)

The `summarize()` method creates concise summaries of longer content, with optional context for focused summarization:

```python
sym_long = Symbol("Python is a high-level, interpreted programming language...")
summarized = sym_long.summarize()
# Creates a shorter version maintaining key information

# With context for focused summarization
context_summarized = sym_long.summarize(context="Focus on Python's use in data science")
# Emphasizes data science aspects in the summary
```

### Content Outlining (`outline`)

The `outline()` method extracts structure and key points from hierarchical content:

```python
sym_complex = Symbol("""
#Introduction to Machine Learning
Machine learning is a subset of artificial intelligence...
## Supervised Learning
### Classification
### Regression
## Unsupervised Learning
""")
outlined = sym_complex.outline()
# => ['- Machine Learning: subset of AI...', '- Supervised Learning: training...', ...]
```

### Content Filtering (`filter`)

The `filter()` method selectively includes or excludes content based on criteria:

```python
sym_mixed = Symbol("Dogs are loyal pets. Cats are independent pets. Hamsters are small pets.")

# Exclude content matching criteria (default behavior)
filtered_ex = sym_mixed.filter(criteria="Cats")
# => "Dogs are loyal pets. Hamsters are small pets."

# Include only content matching criteria
filtered_in = sym_mixed.filter(criteria="Dogs", include=True)
# => "Dogs are loyal pets."
```

### Content Modification (`modify`)

The `modify()` method applies specified changes to content:

```python
sym_original = Symbol("The quick brown fox jumps over the lazy dog.")
changes = "Change 'quick' to 'fast' and 'lazy' to 'sleeping'"
modified = sym_original.modify(changes=changes)
# => "The fast brown fox jumps over the sleeping dog."
```

### Text Replacement (`replace`)

The `replace()` method substitutes specific text with new content:

```python
sym_replace = Symbol("Python is a programming language. Python is easy to learn.")
replaced = sym_replace.replace("Python", "JavaScript")
# => "JavaScript is a programming language. JavaScript is easy to learn."
```

### Content Removal (`remove`)

The `remove()` method eliminates specified content:

```python
sym_extra = Symbol("This text contains [unnecessary information] that should be removed.")
removed = sym_extra.remove("[unnecessary information] ")
# => "This text contains that should be removed."
```

### Content Addition (`include`, `combine`)

Both methods add content, with subtle differences in behavior:

```python
sym_base = Symbol("This is the main content.")

# Include additional information
included = sym_base.include("This is additional information.")
# => "This is the main content. This is additional information."

# Combine with other content
combined = sym_base.combine("Second part of the content.")
# => "This is the main content. Second part of the content."
```

### Semantic Mapping (`map`)

The `map` operation applies semantic transformations to each element in an iterable based on natural language instructions. It preserves the container type and leaves non-matching elements unchanged.

```python
# Transform characters in strings
text = Symbol("hello world")
result = text.map('convert vowels to numbers: a=1, e=2, i=3, o=4, u=5')
# => "h2ll4 w4rld"

# Transform case selectively
caps_text = Symbol("PROGRAMMING")
result = caps_text.map('make consonants lowercase, keep vowels uppercase')
# => "prOgrAmmIng"

# Transform fruits to vegetables, leave animals unchanged
mixed_list = Symbol(['apple', 'banana', 'cherry', 'cat', 'dog'])
result = mixed_list.map('convert all fruits to vegetables')
# => ['carrot', 'broccoli', 'spinach', 'cat', 'dog']

# Work with dictionaries - transforms values, preserves keys
fruit_dict = Symbol({'item1': 'apple', 'item2': 'banana', 'item3': 'cat'})
result = fruit_dict.map('convert fruits to vegetables')
# => {'item1': 'carrot', 'item2': 'broccoli', 'item3': 'cat'}

# Preserve container types
emotions_tuple = Symbol(('happy', 'sad', 'angry'))
weather = emotions_tuple.map('convert emotions to weather')
# => ('sunny', 'rainy', 'stormy')
```

**Supported Types:**
- Strings: `"abc"` → `"xyz"` (character-by-character transformation)
- Lists: `[...]` → `[...]`
- Tuples: `(...)` → `(...)`
- Sets: `{...}` → `{...}`
- Dictionaries: `{key: value}` → `{key: transformed_value}`

**Note:** For strings, transformations apply to individual characters and the result is joined back into a string. For dictionaries, transformations apply to values while preserving keys.
### Pattern Extraction (`extract`)

The `extract()` method identifies and extracts specific patterns or information types:

```python
sym_contact = Symbol("""
Contact Information:
Email: john.doe@email.com
Phone: +1-555-0123
Address: 123 Main St, City
""")
extracted = sym_contact.extract("contact details")
# => "Email: john.doe@email.com | Phone: +1-555-0123 | Address: 123 Main St, City"

# Extract specific patterns like dates
sym_text = Symbol("""
Project deadline: 2024-03-15
Budget allocated: $50,000
Team size: 8 people
Status: In Progress
""")
extracted_dates = sym_text.extract("dates and deadlines")
# => "2024-03-15"
```

### Code Correction (`correct`)

The `correct()` method automatically fixes code errors based on exceptions:

```python
# Fix syntax errors
sym_code = Symbol("""
def calculate_sum(a b):
    return a + b
""")
corrected = sym_code.correct("Fix the code", exception=SyntaxError)
# => "def calculate_sum(a, b):\n    return a + b"

# Fix type errors
sym_type_error = Symbol("""
def process_data(items):
    return items.sort()
result = process_data([3, 1, 2])
print(result + 1) # TypeError: NoneType + int
""")
corrected_type = sym_type_error.correct("Fix the code", exception=TypeError)
# => Uses sorted() instead of sort() to return the sorted list
```

### Language Translation (`translate`)

The `translate()` method converts content between languages with optional formality control:

```python
sym_english = Symbol("Hello, how are you today?")
translated = sym_english.translate("Spanish")
# => "Hola, ¿cómo estás hoy?"

# Formal translation
sym_informal = Symbol("Hey there! What's up?")
translated_formal = sym_informal.translate("French", formal=True)
# => "Salut ! Quoi de neuf ?" (Note: maintains appropriate formality)
```

### Multi-Choice Classification (`choice`)

The `choice()` method classifies content into predefined categories:

```python
sym_weather = Symbol("Temperature: 85°F, Humidity: 70%, Conditions: Sunny")
cases = ["hot and humid", "mild", "cold and dry"]
weather_choice = sym_weather.choice(cases=cases, default="mild")
# => "hot and humid"

# Sentiment analysis
sym_sentiment = Symbol("This product exceeded all my expectations! Absolutely wonderful!")
sentiment_cases = ["positive", "neutral", "negative"]
sentiment_choice = sym_sentiment.choice(cases=sentiment_cases, default="neutral")
# => "positive"
```

### Format Conversion (`convert`)

The `convert()` method transforms data between different formats:

```python
sym_json = Symbol('{"name": "John", "age": 30, "city": "New York"}')

yaml_converted = sym_json.convert("YAML")
# => "name: John\nage: 30\ncity: New York"

xml_converted = sym_json.convert("XML")
# => "<data>\n  <name>John</name>\n  <age>30</age>\n  <city>New York</city>\n</data>"

# Convert CSV to HTML table
sym_csv = Symbol("Name,Age,Department\nAlice,28,Engineering\nBob,35,Marketing")
table_converted = sym_csv.convert("HTML table")
# => Complete HTML table with thead and tbody
```

### Content Transcription (`transcribe`)

The `transcribe()` method modifies content style, tone, and formality:

```python
sym_informal = Symbol("Hey there! How's it going? Hope you're doing well!")
formal_transcribed = sym_informal.transcribe("make it formal and professional")
# => "I hope this message finds you well."

sym_technical = Symbol("The system crashed because of insufficient memory allocation.")
simple_transcribed = sym_technical.transcribe("explain in simple terms for non-technical audience")
# => "The system stopped working because it didn't have enough digital space to run properly."
```

## 15. Execution Control and Code Operations

### Code Analysis (`analyze`)

The `analyze()` method provides insights into code issues and exceptions:

```python
sym_code = Symbol("print('Hello World')")
try:
    raise ValueError("Sample error for testing")
except Exception as e:
    analyzed = sym_code.analyze(exception=e, query="What went wrong?")
# => Detailed analysis of the error context
```

### Code Execution (`execute`, `fexecute`)

Execute Python code and return comprehensive results:

```python
sym_code = Symbol("""
def run():
    return 2 + 3
res = run()
""")
executed = sym_code.execute()
# => {'globals': {...}, 'locals': {...}, 'locals_res': 5}

# Fallback execution
fexecuted = sym_code.fexecute()
# => Similar execution with fault tolerance
```

### Code Simulation (`simulate`)

The `simulate()` method provides step-by-step execution traces:

```python
sym_algorithm = Symbol("x = 5; y = 10; result = x + y")
simulated = sym_algorithm.simulate()
# => "Step 4: Set result = 15"

# Complex algorithm simulation
sym_fibonacci = Symbol("def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)")
algorithm_sim = sym_fibonacci.simulate()
# => Detailed pseudocode execution trace
```

### Information Sufficiency (`sufficient`)

Check if provided information is adequate for specific purposes:

```python
sym_info = Symbol("Product Name: UltraBook Pro\nPrice: $1299\nRAM: 16GB")
sufficient_check = sym_info.sufficient("Is this enough information to make a purchase decision?")
# => False (missing key details)
```

### List Processing (`list`, `foreach`)

Extract and process lists based on criteria:

```python
sym_employees = Symbol("""
Bob - Manager - $85000
Carol - Designer - $65000
David - Engineer - $72000
""")

engineers_list = sym_employees.list("employees who are engineers")
# => ['David']

# Apply operations to each item
sym_numbers = Symbol([1, 2, 3, 4, 5])
squared_numbers = sym_numbers.foreach("each number", "square the number")
# => ['1', '4', '9', '16', '25']
```

### Fault-Tolerant Execution (`ftry`)

Execute operations with retry logic and error handling:

```python
sym_test = Symbol("test data")
# Automatically retries failed operations
ftry_result = sym_test.ftry(expression, retries=2)
# => Successful execution after potential retries
```

## 16. Dictionary Handling Operations

### Dictionary Creation (`dict`)

Convert various data types into structured dictionaries:

```python
sym_text = Symbol("I have apples, oranges, and bananas in my kitchen.")
dict_result = sym_text.dict("categorize fruits")
# => {'fruits': ['apples', 'oranges', 'bananas'], 'description': ['I have', 'in my kitchen.']}

sym_shopping = Symbol("milk, bread, eggs, apples, cheese, chicken, broccoli, rice")
food_dict = sym_shopping.dict("organize by food categories")
# => {'dairy': ['milk', 'cheese'], 'produce': ['apples', 'broccoli'], ...}
```

## 17. Template and Styling Operations

### Template Substitution (`template`)

Replace placeholders in templates with Symbol values:

```python
sym_name = Symbol("Alice")
template_str = "Hello {{placeholder}}, welcome to our service!"
templated_result = sym_name.template(template_str)
# => "Hello Alice, welcome to our service!"

# Custom placeholder
sym_product = Symbol("Premium Headphones")
custom_template = "Product: ***ITEM*** - Now available!"
custom_templated = sym_product.template(custom_template, placeholder="***ITEM***")
# => "Product: Premium Headphones - Now available!"
```

### Content Styling (`style`)

Apply formatting and styling to content:

```python
sym_content = Symbol("This is basic text that needs styling.")
styled_result = sym_content.style("Make this text bold and italic")
# => "***This is basic text that needs styling.***"

sym_code = Symbol("print('Hello World')")
styled_code = sym_code.style("Format as Python code with syntax highlighting", libraries=["python"])
# => "```python\nprint('Hello World')\n```"
```

## 18. Data Clustering Operations

### Semantic Clustering (`cluster`)

Group similar items based on semantic similarity:

```python
sym_fruits = Symbol(["apple", "banana", "cherry", "orange", "grape"])
clustered_result = sym_fruits.cluster(metric='cosine', min_cluster_size=2)
# => Groups semantically similar items together

sym_sentences = Symbol([
    "The weather is sunny today.",
    "It's raining heavily outside.",
    "Beautiful sunny weather today."
])
sentence_clusters = sym_sentences.cluster(min_cluster_size=2)
# => Groups similar sentiment/topic sentences
```

Under the hood, the `cluster` method uses `sklearn`'s HDBSCAN [algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN). You can pass as `kwargs` any parameter that is accepted by the `HDBSCAN` class, such as `min_samples`, `min_cluster_size`, `metric`, etc.

## 19. Embedding and Similarity Operations

### Embedding Generation (`embed`, `embedding`)

Create vector embeddings for semantic operations:

```python
sym_text = Symbol("hello world")
embedded_result = sym_text.embed()
# => Embedding vectors with shape information

# Access embedding property
embedding_value = sym_text.embedding
# => NumPy array of embeddings
```

### Similarity and Distance Calculations

Calculate semantic similarity and distance between symbols:

```python
sym1 = Symbol("hello world")
sym2 = Symbol("hi there")

similarity_result = sym1.similarity(sym2, metric='cosine')
distance_result = sym1.distance(sym2, kernel='gaussian')
```

### Embedding Packaging (`zip`)

Format embeddings for storage and retrieval:

```python
sym_text = Symbol("hello world")
zip_result = sym_text.zip()
# => [(id, embedding_list, query_dict), ...]
```

## 20. Input/Output Handling Operations

### File Operations (`open`)

Read file contents into Symbol values:

```python
# Open file with path parameter
sym_empty = Symbol()
opened_result = sym_empty.open("/path/to/file.txt")
# => Symbol containing file contents

# Open file with path as Symbol value
sym_path = Symbol("/path/to/file.txt")
opened_from_value = sym_path.open()
# => Symbol containing file contents
```

### User Input (`input`)

Handle interactive user input (requires user interaction):

```python
sym = Symbol()
user_input = sym.input() # Prompts for user input
```

## 21. Persistence Operations

### Saving Symbols (`save`)

Persist Symbol data to files with various options:

```python
sym_test = Symbol("Hello, persistence test!")

# Save as pickle (serialized)
sym_test.save("/path/to/file.pkl", serialize=True)

# Save as text
sym_test.save("/path/to/file.txt", serialize=False)

# Control file replacement
sym_test.save("/path/to/file.txt", replace=False) # Creates new file with suffix
sym_test.save("/path/to/file.txt", replace=True)  # Overwrites existing file
```

### Loading Symbols (`load`)

Load previously saved Symbol data:

```python
loader_sym = Symbol()
loaded_sym = loader_sym.load("/path/to/file.pkl")
# => Restored Symbol with original data
```

### Dynamic Function Expansion (`expand`)

Generate and attach new functions based on descriptions:

```python
sym_task = Symbol("Calculate the fibonacci sequence up to 10 numbers")
func_name = sym_task.expand()
# => Generated function name, function attached as Symbol attribute
```

## 22. Output Processing Operations

### Output Handling (`output`)

Process and format Symbol output with custom handlers:

```python
sym_test = Symbol("Hello, output test!")

# Basic output processing
result = sym_test.output()
# => Structured output with processed information

# Custom handler function
def custom_handler(input_dict):
   # Process the input dictionary
    pass

result_with_handler = sym_test.output(handler=custom_handler)
# => Output processed through custom handler

# With additional arguments
result_with_args = sym_test.output("arg1", "arg2", custom_param="value")
# => Output including method arguments and parameters
```
