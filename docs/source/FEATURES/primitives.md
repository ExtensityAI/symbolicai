# Primitives Cheat-Sheet

> *“Primitives are the atoms of the SymbolicAI API – small, orthogonal operations that can be freely
> combined to express very rich behaviour.  Every `Symbol` automatically inherits them, so there is no
> import-gymnastics required: just create a symbol and start hacking.”*

This page gives you a **hands-on overview** of the most useful primitives that ship with
`symai`. All examples are *copy-pastable* – open a Python REPL and play with them!  The table of
contents below mirrors the internal organisation of `symai/ops/primitives.py`, so if you ever need
more detail you know where to look.

---
## 1. Syntactic **vs.** Semantic Symbols 🧭

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

   s = Symbol("Cats are adorable", semantic=True)   # already semantic
   ```

2. **On demand with the `.sem` projection** – the twin `.syn` flips you back:

   ```python
   s = Symbol("Cats are adorable")   # default = syntactic

   print(s.startswith("Cats"))   # syntactic → True

   print(s.sem.contains("animals"))  # semantic  → True

   # You can chain the projections like brush-strokes on a canvas:
   result = s.sem.split(" ").syn[0]   # semantic split, syntactic index
   print(result)  # 'Cats'
   ```

Because the projections return the *same underlying object* with just a different behavioural
coat, you can weave complex chains of syntactic and semantic operations on a single symbol – **be
like an artist and paint with both modes!**

Behind the scenes the semantic path sends your query to the configured neuro-symbolic engine while
the syntactic path keeps everything local and lightning-fast.

---

## 2. Operator Primitives  🧮

These primitives piggy-back on Python’s operator overloading.  When you see them in code they look
and feel *exactly* like the regular operators you already know.

| Operator | Works On | Example (syntactic) | Example (semantic) |
|----------|----------|---------------------|--------------------|
| `a == b` | any      | `Symbol(42) == 42              # True` | `Symbol("Hi") .sem == "Hello"   # True – greetings are equal` |
| `a != b` | any      | `Symbol("foo") != "bar"       # True` | `Symbol("sunny") .sem != "rainy" # True – opposite weather` |
| `>` / `<`/`>=`/`<=` | numbers, strings | `Symbol(10) > 5` | `Symbol("elephant").sem < "mouse"  # size semantics` |
| `~a`     | bool / str / int | `~Symbol(True)        # False` | `~Symbol("I am happy").sem  # "I am sad."` |
| `a & b`  | bool / str | `Symbol(True) & False      # False` | `Symbol("All birds fly").sem & "Penguins are birds"` |
| `a \| b`  | bool / str | `Symbol(False) \| True      # True` | `Symbol("It may rain").sem \| "It may snow"` |
| `a << b` | int / str | `Symbol(3) << 2            # 12` | `Symbol("urgent").sem << "task"   # "urgent task"` |
| `a >> b` | int / str | `Symbol(8) >> 1            # 4`  | `Symbol("pipeline").sem >> "step" # "pipeline step"` |

👉 **Tip:** The usual inplace variants (`+=`, `&=`, `<<=`, …) are also available.

---

## 3. Casting Primitives  🔀

Turn a symbol into a different Python type **without** losing provenance:

```python
s = Symbol("123.45")

print(s.int())        # 123
print(s.float())      # 123.45
print(s.bool())       # True
print(s.str())        # "123.45"

# Safely evaluate Python literals (similar to ast.literal_eval)
lst = Symbol("[1, 2, 3]").ast()   # → [1, 2, 3]

# Seamless view switching
print(s.sem)          # semantic view of the same value
print(s.syn)          # syntactic view (default if you were already semantic)
```

---

## 4. Iteration Primitives  🔢

Work with symbols like containers – indexing, slicing, setting and deleting items all feel
natural:

```python
seq  = Symbol(["🍎", "🍌", "🍒", "🍇"])

print(seq[1])          # "🍌"

seq[2] = "🥝"          # mutate in-place
del seq[0]             # remove first element

print(seq.value)       # ["🍌", "🥝", "🍇"]

# Semantic lookup (falls back to the engine when the key is fuzzy)
colors = Symbol({"red": "#FF0000", "green": "#00FF00", "blue": "#0000FF"})
print(colors.sem["primary color"])   # ["#FF0000", "#00FF00", "#0000FF"]
```

---

## 5. String Helper Primitives  🧩

Handy utilities for everyday text-crunching:

```python
txt = Symbol("hello world python programming")

words = txt.split(" ")            # → [Symbol("hello"), Symbol("world"), ...]
print(words.join("-"))            # hello-world-python-programming

print(txt.startswith("hello"))    # True
print(txt.endswith("ing"))        # True

# Semantic flavour
print(txt.sem.startswith("greeting"))   # True
```

---

## 6. Advanced Comparison Primitives  🔍

When a simple operator is not enough you can drop down to explicit methods that accept extra
context:

```python
quote   = Symbol("The quick brown fox jumps over the lazy dog")

# Context-aware equality
print(quote.equals("A nimble dark fox leaps above a tired dog", context="animal story"))

# Fuzzy containment (works on any value, not just strings)
print(quote.contains("fox"))              # True

# Semantic type checks
print(Symbol(3.14).isinstanceof("number"))        # True
print(Symbol({"name": "Alice"}).isinstanceof("person data"))  # True
```

---

## 7. Expression Handling  🧠

`interpret` turns a plain string into a *computable expression* – think of it as a built-in mini
CoT (chain-of-thought) executor:

```python
from math import tau

# 1. Natural language Q&A
answer = Symbol("What is the tallest mountain in the world?").interpret()
print(answer)   # Mount Everest

# 2. Analogies
print(Symbol("gravity : Earth :: radiation : ?").interpret())

# 3. Inline calculus
expr = Symbol("∫(3x² + 2x − 5)dx").interpret()
print(expr)     # x³ + x² − 5x + C

# Keep track of multiple calls
sym = Symbol("Propose a hypothesis about climate change")
sym.interpret(accumulate=True)
sym.interpret("Provide evidence", accumulate=True)
print(sym.get_results())  # list with both answers
sym.clear_results()
```

---

## 8. Advanced Chaining Example 🚀

This example demonstrates the true power of SymbolicAI primitives—a sophisticated chain that seamlessly weaves syntactic and semantic operations together, showcasing text processing, sentiment analysis, data transformation, and intelligent clustering:

```python
from symai import Symbol
import numpy as np

# Start with customer feedback data
reviews = Symbol([
    "This product is absolutely amazing! Best purchase ever.",
    "Terrible quality, broke after one day. Very disappointed.",
    "Good value for money, works as expected.",
    "Outstanding customer service and fast delivery!",
    "Poor design, difficult to use. Would not recommend.",
    "Excellent build quality, very satisfied with purchase."
])

# 1. Semantic sentiment analysis with syntactic indexing
sentiments = (reviews.sem
              .foreach("review", "classify sentiment as positive/negative/neutral")
              .syn)  # Switch to syntactic for fast operations

# 2. Syntactic filtering + semantic enhancement
positive_reviews = (sentiments.syn
                   .split(", ")  # Split the results
                   .sem
                   .filter("positive sentiment", include=True)
                   .syn)

# 3. Complex semantic-syntactic chain: extract keywords and organize
keywords_dict = (reviews.sem
                .foreach("review", "extract 3 key product aspects mentioned")
                .syn
                .join(" | ")  # Syntactic join
                .sem
                .dict("organize by product feature categories"))

# 4. Semantic similarity with syntactic membership testing
sample_review = Symbol("Great product, highly recommended!")
similar_reviews = []

for i, review in enumerate(reviews.value):
    review_sym = Symbol(review)
    # Semantic similarity check
    if review_sym.sem.similarity(sample_review) > 0.7:
        similar_reviews.append(f"Review {i+1}")

# 5. Advanced text transformation chain
summary = (reviews.sem
          .summarize("focus on product quality and user experience")
          .syn
          .replace("product", "item")  # Syntactic replacement
          .sem
          .transcribe("make more professional and concise")
          .syn
          .template("SUMMARY: {{placeholder}}")  # Syntactic templating
          .sem)

# 6. Semantic comparison with syntactic operators
quality_mentions = Symbol("The build quality is exceptional")
service_mentions = Symbol("Customer service was outstanding")

# Semantic comparison using overloaded operators
if quality_mentions.sem > service_mentions.sem:
    priority = "quality"
else:
    priority = "service"

# 7. Data clustering with embedding extraction
if len(reviews.value) > 1:
    # Extract embeddings semantically, process syntactically
    embeddings = [Symbol(review).sem.embedding.flatten() for review in reviews.value]
    
    # Syntactic numpy operations
    X = np.array(embeddings)
    
    # Simple clustering based on similarity threshold
    clusters = {}
    cluster_id = 0
    processed = set()
    
    for i, emb1 in enumerate(embeddings):
        if i in processed:
            continue
            
        cluster = [i]
        processed.add(i)
        
        for j, emb2 in enumerate(embeddings[i+1:], i+1):
            if j not in processed:
                # Semantic similarity check
                sim = Symbol(reviews.value[i]).sem.similarity(Symbol(reviews.value[j]))
                if sim > 0.8:  # High similarity threshold
                    cluster.append(j)
                    processed.add(j)
        
        clusters[cluster_id] = [reviews.value[idx] for idx in cluster]
        cluster_id += 1

# 8. Final semantic synthesis with syntactic formatting
final_analysis = (Symbol(f"Priority: {priority}")
                 .sem
                 .combine(str(summary))
                 .syn
                 .template("ANALYSIS REPORT:\n{{placeholder}}")
                 .sem
                 .style("format as professional business report"))

print("🎯 Sentiment Analysis:", sentiments.value)
print("✨ Keywords by Category:", keywords_dict.value)
print("📊 Similar Reviews:", similar_reviews)
print("🔍 Review Clusters:", len(clusters), "groups found")
print("📋 Final Analysis:")
print(final_analysis.value)
```

This example showcases:
- **Seamless `.syn` ↔ `.sem` switching** for optimal performance
- **Operator overloading** (`>`, `in`) working semantically
- **Complex chaining** of 15+ different primitives
- **Real-world application** (customer feedback analysis)
- **Mixed data types** (strings, lists, embeddings, dictionaries)
- **Template and styling** for professional output formatting
- **Intelligent clustering** based on semantic similarity

The beauty lies in how syntactic operations (fast, local) and semantic operations (intelligent, contextual) flow together naturally, creating powerful data processing pipelines that are both efficient and intelligent.

## 9. Quick Reference Table

| Category | Primitive | Signature | Semantics |
|----------|-----------|-----------|-----------|
| Casting  | `str()` `int()` `float()` `bool()` `ast()` | `Symbol -> T` | Fast local cast, raises on failure |
| String   | `split(delim)` `join(delim=" ")` | text <-> list | Pure-Python for syntactic, engine for semantic |
| Compare  | `equals(other, context="…")` `contains(item)` `isinstanceof(type_desc)` | bool | Semantic by default |
| Iterate  | `obj[key]`, `obj[key]=`, `del obj[key]` | container ops | Fuzzy key support in semantic mode |
| Logical  | `&` `|` `~` | conjunction, disjunction, negation | Works on bool and natural language |
| Bitshift | `<<` `>>` | shift / prepend-append semantics | Strings & ints |
| Maths    | `+` `-` `*` `/` `**` `%` `//` | standard arithmetics | Extended to symbolic maths |

Feel free to explore the full API – any `dir(Symbol())` call will reveal even more gems ✨.
