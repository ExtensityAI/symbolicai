# Expressions

---

## ⚠️  Outdated or Deprecated Documentation ⚠️
This documentation is outdated and may not reflect the current state of the SymbolicAI library. This page might be revived or deleted entirely as we continue our development. We recommend using more modern tools that infer the documentation from the code itself, such as [DeepWiki](https://deepwiki.com/ExtensityAI/symbolicai). This will ensure you have the most accurate and up-to-date information and give you a better picture of the current state of the library.

---

## Overview
An `Expression` is a non-terminal symbol that can be further evaluated. It inherits all the properties from the Symbol class and overrides the `__call__` method to evaluate its expressions or values. All other expressions are derived from the `Expression` class, which also adds additional capabilities, such as the ability to `fetch` data from URLs, `search` on the internet, or `open` files. These operations are specifically separated from the `Symbol` class as they do not use the `value` attribute of the Symbol class.

## Expression Design
SymbolicAI's API closely follows best practices and ideas from `PyTorch`, allowing the creation of complex expressions by combining multiple expressions as a computational graph. Each Expression has its own `forward` method that needs to be overridden. The `forward` method is used to define the behavior of the expression. It is called by the `__call__` method, which is inherited from the `Expression` base class. The `__call__` method evaluates an expression and returns the result from the implemented `forward` method. This design pattern evaluates expressions in a lazy manner, meaning the expression is only evaluated when its result is needed. It is an essential feature that allows us to chain complex expressions together. Numerous helpful expressions can be imported from the `symai.components` file.

## Core Properties
Other important properties inherited from the Symbol class include `sym_return_type` and `static_context`. These two properties define the context in which the current Expression operates, as described in the [Prompt Design](operations.md#prompt-design) section. The `static_context` influences all operations of the current Expression sub-class. The `sym_return_type` ensures that after evaluating an Expression, we obtain the desired return object type. It is usually implemented to return the current type but can be set to return a different type.

## Expression Structure
Expressions may have more complex structures and can be further sub-classed, as shown in the `Sequence` expression example in the following figure:

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img2.png" width="720px">

A Sequence expression can hold multiple expressions evaluated at runtime.

## Expression Types

### Sequence Expressions
Here is an example of defining a Sequence expression:

```python
# First import all expressions
from symai.components import *
# Define a sequence of expressions
Sequence(
    Clean(),
    Translate(),
    Outline(),
    Compose('Compose news:'),
)
```

### Stream expressions

As previously mentioned, we can create contextualized prompts to define the behavior of operations on our neural engine. However, this limits the available context size due to GPT-3 Davinci's context length constraint of 4097 tokens. This issue can be addressed using the `Stream` processing expression, which opens a data stream and performs chunk-based operations on the input stream.

A Stream expression can be wrapped around other expressions. For example, the chunks can be processed with a `Sequence` expression that allows multiple chained operations in a sequential manner. Here is an example of defining a Stream expression:

```python
Stream(Sequence(
    Clean(),
    Translate(),
    Outline(),
    Embed()
))
```

The example above opens a stream, passes a `Sequence` object which cleans, translates, outlines, and embeds the input. Internally, the stream operation estimates the available model context size and breaks the long input text into smaller chunks, which are passed to the inner expression. The returned object type is a `generator`.

This approach has the drawback of processing chunks independently, meaning there is no shared context or information among chunks. To address this issue, the `Cluster` expression can be used, where the independent chunks are merged based on their similarity, as illustrated in the following figure:

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img6.png" width="720px">

In the illustrated example, all individual chunks are merged by clustering the information within each chunk. It consolidates contextually related information, merging them meaningfully. The clustered information can then be labeled by streaming through the content of each cluster and extracting the most relevant labels, providing interpretable node summaries.

The full example is shown below:

```python
stream = Stream(Sequence(
    Clean(),
    Translate(),
    Outline(),
))
sym = Symbol('<some long text>')
res = Symbol(list(stream(sym)))
expr = Cluster()
expr(res)
```

Next, we could recursively repeat this process on each summary node, building a hierarchical clustering structure. Since each Node resembles a summarized subset of the original information, we can use the summary as an index. The resulting tree can then be used to navigate and retrieve the original information, transforming the large data stream problem into a search problem.

Alternatively, vector-based similarity search can be used to find similar nodes. Libraries such as [Annoy](https://github.com/spotify/annoy), [Faiss](https://github.com/facebookresearch/faiss), or [Milvus](https://github.com/milvus-io/milvus) can be employed for searching in a vector space.
