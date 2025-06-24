# ❌ Error Handling and Debugging

---

## ⚠️  Outdated or Deprecated Documentation ⚠️
This documentation is outdated and may not reflect the current state of the SymbolicAI library. This page might be revived or deleted entirely as we continue our development. We recommend using more modern tools that infer the documentation from the code itself, such as [DeepWiki](https://deepwiki.com/ExtensityAI/symbolicai). This will ensure you have the most accurate and up-to-date information and give you a better picture of the current state of the library.

---

A key idea of the SymbolicAI API is code generation, which may result in errors that need to be handled contextually. In the future, we want our API to self-extend and resolve issues automatically. We propose the `Try` expression, which has built-in fallback statements and retries an execution with dedicated error analysis and correction. The expression analyzes the input and error, conditioning itself to resolve the error by manipulating the original code. If the fallback expression succeeds, the result is returned. Otherwise, this process is repeated for the specified number of `retries`. If the maximum number of retries is reached and the problem remains unresolved, the error is raised again.

Suppose we have some executable code generated previously. By the nature of generative processes, syntax errors may occur. Using the `Execute` expression, we can evaluate our generated code, which takes in a symbol and tries to execute it. Naturally, this will fail. However, in the following example, the `Try` expression resolves the syntax error, and we receive a computed and corrected result.

```python
expr = Try(expr=Execute())
sym = Symbol('a = int("3,")') # Some code with a syntax error
res = expr(sym)
print(res) # Output: a = 3
```

We are aware that not all errors are as simple as the syntax error example shown, which can be resolved automatically. Many errors occur due to semantic misconceptions, requiring contextual information. We are exploring more sophisticated error handling mechanisms, including the use of streams and clustering to resolve errors in a hierarchical, contextual manner. It is also important to note that neural computation engines need further improvements to better detect and resolve errors.

## 🕷️ Interpretability, Testing & Debugging

Perhaps one of the most significant advantages of using neuro-symbolic programming is that it allows for a clear understanding of how well our LLMs comprehend simple operations. Specifically, we gain insight into whether and at what point they fail, enabling us to follow their StackTraces and pinpoint the failure points. In our case, neuro-symbolic programming enables us to debug the model predictions based on dedicated unit tests for simple operations. To detect conceptual misalignments, we can use a chain of neuro-symbolic operations and validate the generative process. Although not a perfect solution, as the verification might also be error-prone, it provides a principled way to detect conceptual flaws and biases in our LLMs.

### Unit Testing Models

Since our approach is to divide and conquer complex problems, we can create conceptual unit tests and target very specific and tractable sub-problems. The resulting measure, i.e., the success rate of the model prediction, can then be used to evaluate their performance and hint at undesired flaws or biases.

This method allows us to design domain-specific benchmarks and examine how well general learners, such as GPT-3, adapt with certain prompts to a set of tasks.

For example, we can write a fuzzy comparison operation that can take in digits and strings alike and perform a semantic comparison. LLMs can then be asked to evaluate these expressions. Often, these LLMs still fail to understand the semantic equivalence of tokens in digits vs. strings and provide incorrect answers.

The following code snippet shows a unit test to perform semantic comparison of numbers (between digits and strings):

```python
import unittest
from symai import *

class TestComposition(unittest.TestCase):
  def test_compare(self):
      res = Symbol(10) > Symbol(5)
      self.assertTrue(res)
      res = Symbol(1) < Symbol('five')
      self.assertTrue(res)
      ...
```

### 🔥Debugging

When creating complex expressions, we debug them by using the `Trace` expression, which allows us to print out the applied expressions and follow the StackTrace of the neuro-symbolic operations. Combined with the `Log` expression, which creates a dump of all prompts and results to a log file, we can analyze where our models potentially failed.

#### Example: News Summary

In the following example, we create a news summary expression that crawls the given URL and streams the site content through multiple expressions. The outcome is a news website created based on the crawled content. The `Trace` expression allows us to follow the StackTrace of the operations and observe which operations are currently being executed. If we open the `outputs/engine.log` file, we can see the dumped traces with all the prompts and results.

```python
# crawling the website and creating a website based on its facts
news = News(url='https://www.cnbc.com/cybersecurity/',
            pattern='cnbc',
            filters=ExcludeFilter('sentences about subscriptions, licensing, newsletter'),
            render=True)
expr = Log(Trace(news))
res = expr()
```

See {doc}`here <../TUTORIALS/context>` for more information and full news summary code.

Here is the corresponding StackTrace of the model:

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img8.png" width="900px">

## 📈 Interface for Query and Response Inspection

SymbolicAI is a data-driven framework by design. This implies that we can gather data from API interactions while delivering the requested responses. For rapid, dynamic adaptations or prototyping, we can swiftly integrate user-desired behavior into existing prompts. Moreover, we can log user queries and model predictions to make them accessible for post-processing. Consequently, we can enhance and tailor the model's responses based on real-world data.

In the example below, we demonstrate how to use an `Output` expression to pass a handler function and access the model's input prompts and predictions. These can be utilized for data collection and subsequent fine-tuning stages. The handler function supplies a dictionary and presents keys for `input` and `output` values. The content can then be sent to a data pipeline for additional processing.

```python
sym = Symbol('Hello World!')
def handler(res):
    input_ = res['input']
    output = res['output']
expr = Output(expr=sym.translate,
              handler=handler,
              verbose=True)
res = expr('German')
```

Since we used verbose, the console print of the `Output` expression is also visible:

```bash
Input: (['Translate the following text into German:\n\nHello World!'],)
Expression: <bound method Symbol.translate of <class 'symai.symbol.Symbol'>(value=Hello World!)>
args: ('German',) kwargs: {'input_handler': <function OutputEngine.forward.<locals>.input_handler at ...
Dictionary: {'instance': <class 'symai.components.Output'>(value=None), 'func': <function Symbol.output.<locals>._func at ...
Output: Hallo Welt!
```
