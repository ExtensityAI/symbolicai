# Quick Start Guide

This guide will help you get started with SymbolicAI, demonstrating basic usage and key features.

**Note**: the most accurate documentation is the _code_, so be sure to check out the tests. Look for the `mandatory` mark since those are the features that were tested and are guaranteed to work.

For more detailed information about usage, refer to the materials available [here](https://extensityai.gitbook.io/symbolicai/features/contracts) and [here](https://extensityai.gitbook.io/symbolicai/features/primitives).

To start, import the library by using:

```python
from symai import Symbol
```

## Creating and Manipulating Symbols

Our `Symbolic API` is based on object-oriented and compositional design patterns. The `Symbol` class serves as the base class for all functional operations, and in the context of symbolic programming (fully resolved expressions), we refer to it as a terminal symbol. The Symbol class contains helpful operations that can be interpreted as expressions to manipulate its content and evaluate new Symbols `<class 'symai.expressions.Symbol'>`.

```python
# Create a Symbol
S = Symbol("Welcome to our tutorial.")
# Translate the Symbol
print(S.translate('German')) # Output: Willkommen zu unserem Tutorial.
```

### Ranking Objects

Our API can also execute basic data-agnostic operations like `filter`, `rank`, or `extract` patterns. For instance, we can rank a list of numbers:

```python
# Ranking objects
import numpy as np

S = Symbol(np.array([1, 2, 3, 4, 5, 6, 7]))
print(S.rank(measure='numerical', order='descending')) # Output: ['7', '6', '5', '4', '3', '2', '1']
```

### Evaluating Expressions

Evaluations are resolved in the language domain and by best effort. We showcase this on the example of [word2vec](https://arxiv.org/abs/1301.3781).

**Word2Vec** generates dense vector representations of words by training a shallow neural network to predict a word based on its neighbors in a text corpus. These resulting vectors are then employed in numerous natural language processing applications, such as sentiment analysis, text classification, and clustering.

In the example below, we can observe how operations on word embeddings (colored boxes) are performed. Words are tokenized and mapped to a vector space where semantic operations can be executed using vector arithmetic.

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img3.png" width="450px">

Similar to word2vec, we aim to perform contextualized operations on different symbols. However, as opposed to operating in vector space, we work in the natural language domain. This provides us the ability to perform arithmetic on words, sentences, paragraphs, etc., and verify the results in a human-readable format.

The following examples display how to evaluate such an expression using a string representation:

```python
# Word analogy
S = Symbol('King - Man + Women').interpret()
print(S)  # Output: Queen
```

### Dynamic Casting

We can also subtract sentences from one another, where our operations condition the neural computation engine to evaluate the Symbols by their best effort. In the subsequent example, it identifies that the word `enemy` is present in the sentence, so it deletes it and replaces it with the word `friend` (which is added):

```python
# Sentence manipulation
S = Symbol('Hello my enemy').sem - 'enemy' + 'friend'
print(S)  # Output: Hello my friend
```

Additionally, the API performs dynamic casting when data types are combined with a Symbol object. If an overloaded operation of the Symbol class is employed, the Symbol class can automatically cast the second object to a Symbol. This is a convenient way to perform operations between `Symbol` objects and other data types, such as strings, integers, floats, lists, etc., without cluttering the syntax.

### Probabilistic Programming

In this example, we perform a fuzzy comparison between two numerical objects. The `Symbol` variant is an approximation of `numpy.pi`. Despite the approximation, the fuzzy equals `==` operation still successfully compares the two values and returns `True`.

```python
# Fuzzy comparison
# # Fuzzy comparison
S = Symbol('3.14159...', semantic=True)
print(S == 'π')  # Output: True
```

### 🧠 Causal Reasoning

The main goal of our framework is to enable reasoning capabilities on top of the statistical inference of Language Models (LMs). As a result, our `Symbol` objects offers operations to perform deductive reasoning expressions. One such operation involves defining rules that describe the causal relationship between symbols. The following example demonstrates how the `&` operator is overloaded to compute the logical implication of two symbols.

```python
S1 = Symbol('The horn only sounds on Sundays.', semantic=True)
S2 = Symbol('I hear the horn.')
res = S1 & S2 # Therefore, it is Sunday.
```

### 🪜 Next Steps
Now, there are tools like DeepWiki that provide better documentation than we could ever write, and we don’t want to compete with that; we'll correct it where it's plain wrong. Please go read SymbolicAI's DeepWiki [page](https://deepwiki.com/ExtensityAI/symbolicai/). There's a lot of interesting stuff in there. Last but not least, check out our [paper](https://arxiv.org/abs/2402.00854) that describes the framework in detail. If you like watching videos, we have a series of tutorials that you can find [here](https://extensityai.gitbook.io/symbolicai/tutorials/video_tutorials).
