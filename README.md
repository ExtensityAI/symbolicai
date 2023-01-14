# <img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/assets/images/bot.png" width="100px"> BotDynamics

## **A Neuro-Symbolic Perspective on Large Language Models (LLMs)**

*Building applications with LLMs at its core through our `Semantic API` leverages the power of classical and differential programming in Python.*

[![PyPI version](https://badge.fury.io/py/botdynamics.svg)](https://badge.fury.io/py/botdynamics) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40DinuMariusC)](https://twitter.com/DinuMariusC) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Xpitfire/botdynamics/issues) ![visitor badge](https://visitor-badge.glitch.me/badge?page_id=botdynamics)

## 📖 Table of Contents

- [ BotDynamics](#-botdynamics)
  - [**A Neuro-Symbolic Perspective on Large Language Models (LLMs)**](#a-neuro-symbolic-perspective-on-large-language-models-llms)
  - [📖 Table of Contents](#-table-of-contents)
  - [🔧 Quick Install](#-quick-install)
    - [➡️ *\[Optional\]* Installs](#️-optional-installs)
  - [🤷‍♂️ Why BotDynamics?](#️-why-botdynamics)
  - [ Tell me some more fun facts!](#-tell-me-some-more-fun-facts)
  - [😶‍🌫️ How does it work?](#️-how-does-it-work)
    - [📚 Symbolic operations](#-symbolic-operations)
    - [Ranking objects](#ranking-objects)
    - [Evaluating Expressions by best effort](#evaluating-expressions-by-best-effort)
    - [Dynamic casting](#dynamic-casting)
    - [Fuzzy Comparisons](#fuzzy-comparisons)
    - [🧠 Causal Reasoning](#-causal-reasoning)
  - [Custom Operations](#custom-operations)
    - [Few-shot operations](#few-shot-operations)
  - [Prompt Design](#prompt-design)
  - [😑 Expressions](#-expressions)
    - [Sequence expressions](#sequence-expressions)
    - [Stream expressions](#stream-expressions)
  - [❌ Error Handling](#-error-handling)
  - [🕷️ Explainability, Testing \& Debugging](#️-explainability-testing--debugging)
    - [Unit Testing Models](#unit-testing-models)
    - [🔥Debugging](#debugging)
    - [Examples](#examples)
  - [📈 Concepts for Data Collection \& Analytics](#-concepts-for-data-collection--analytics)
  - [🤖 Engines](#-engines)
    - [Speech Engine](#speech-engine)
    - [OCR Engine](#ocr-engine)
    - [Search Engine](#search-engine)
    - [WebCrawler Engine](#webcrawler-engine)
    - [Drawing Engine](#drawing-engine)
    - [File Engine](#file-engine)
    - [CLIP Engine](#clip-engine)
  - [⚡Limitations](#limitations)
  - [🥠 Future Work](#-future-work)
  - [Conclusion](#conclusion)
  - [👥 References, Related Work \& Credits](#-references-related-work--credits)
    - [Acknowledgements](#acknowledgements)
    - [Contribution](#contribution)
    - [📜 Citation](#-citation)
    - [📝 License](#-license)
    - [Like this project?](#like-this-project)
    - [📫 Contact](#-contact)



## 🔧 Quick Install

```bash
pip install botdynamics
```

Before the first run, define exports for the required `API keys` to enable the respective engines. This will register the keys in the internal storage. By default BotDynamics currently uses OpenAI's neural engines, i.e. GPT-3 Davinci-003, DALL·E 2 and Embedding Ada-002, for the neuro-symbolic computations, image generation and embeddings computation respectively. However, these modules can easily be replaced with open-source alternatives. Examples are [OPT](https://huggingface.co/docs/transformers/model_doc/opt) or [Bloom](https://huggingface.co/bigscience/bloom) for neuro-symbolic computations, [Craiyon](https://www.craiyon.com/) for image generation, and any [BERT variants](https://huggingface.co/models) for semantic embedding computations. To set the OpenAI API Keys use the following command:

```bash
# Linux / MacOS
export OPENAI_API_KEY="<OPENAI_API_KEY>"

# Windows (PowerShell)
$Env:OPENAI_API_KEY="<OPENAI_API_KEY>"
```


**To get started import our library by using:**

```python
import botdyn as bd
```

Overall, the following engines are currently supported:

* **Neuro-Symbolic Engine**: [OpenAI's LLMs (GPT-3)](https://beta.openai.com/docs/introduction/overview) 
* **Embedding Engine**: [OpenAI's Embedding API](https://beta.openai.com/docs/introduction/overview)
* **[Optional] Search Engine**: [SerpApi](https://serpapi.com/)
* **[Optional] OCR Engine**: [APILayer](https://apilayer.com/ocr)
* **[Optional] SpeechToText Engine**: [OpenAI's Whisper](https://openai.com/blog/whisper/)
* **[Optional] WebCrawler Engine**: [Selenium](https://selenium-python.readthedocs.io/)
* **[Optional] Image Rendering Engine**: [DALL·E 2](https://openai.com/dall-e-2/)
* **[Optional] [CLIP](https://openai.com/blog/clip/) Engine**: 🤗 [HuggingFace](https://huggingface.co/) (experimental image and text embeddings)


### ➡️ *[Optional]* Installs

BotDynamics uses multiple engines to process text, speech and images. We also include search engine access to retrieve information from the web. To use all of them, you will need to install also the following dependencies or assign the API keys to the respective engines. 

If you want to use the `Search Engine` and `OCR Engine` you will need to export the following API keys:

```bash
# Linux / MacOS
export SEARCH_ENGINE_API_KEY="<SERP_API_KEY>"
export OCR_ENGINE_API_KEY="<APILAYER_API_KEY>"

# Windows (PowerShell)
$Env:SEARCH_ENGINE_API_KEY="<SERP_API_KEY>"
$Env:OCR_ENGINE_API_KEY="<APILAYER_API_KEY>"
```

To use them, you will also need to install the following dependencies:


* **SpeechToText Engine**: `ffmpeg` for audio processing (based on OpenAI's [whisper](https://openai.com/blog/whisper/))

```bash
# Linux
sudo apt update && sudo apt install ffmpeg

# MacOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

Additionally, you need to install the newest version directly from their repository, since the version available via `pip` is outdated:

```bash
pip install git+https://github.com/openai/whisper.git
```

* **WebCrawler Engine**: Download for `selenium` the corresponding driver version by setting the `SELENIUM_CHROME_DRIVER_VERSION` environment variable. Currently we use Chrome as the default browser. This means that the Chrome version major number must match the ChromeDriver version. All versions are available [here](https://chromedriver.chromium.org/downloads). For example, if you use chrome version `109.0.5414.74`, you can set any `109.x.x.x` version for the `chromedriver`. In this case the `109.0.5414.74` is available on the selenium page and is for the environment variable available:

```bash
# Linux / MacOS
export SELENIUM_CHROME_DRIVER_VERSION="109.0.5414.74"

# Windows (PowerShell)
$Env:SELENIUM_CHROME_DRIVER_VERSION="109.0.5414.74"
```


## 🤷‍♂️ Why BotDynamics?

BotDynamics tries to close the gap between classical programming or Software 1.0 and modern data-driven development (aka Software 2.0). It is a framework that allows us to build applications with large language models (LLMs) through composability and inheritance - two powerful concepts from object-oriented programming paradigm. However, leveraging ideas from differential programming in Python.

This allows us now to move along a spectrum between these two worlds as illustrated in the following figure:

<img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/assets/images/img5.png" width="720px">

Conceptually, BotDynamics is a framework that uses machine learning - and specifically LLMs - at its core, and curates operations based on dedicated zero or few-shot learning prompt designs. Each operation solves atomic tasks, however, by chaining these operations together we can solve more complex problems. Our main philosophy is to divide and conquer a complex problem into manageable smaller problems. 

In this turn, we also aim to gradually transition between general purpose LLMs with zero and few-shot learning capabilities, and specialized fine-tuned models to really nail down a specific problem (see above). 

## <img src="https://media.giphy.com/media/mGcNjsfWAjY5AEZNw6/giphy.gif" width="50"> Tell me some more fun facts!

In its essence, BotDynamics was inspired by the [`neuro-symbolic programming paradigm`](https://arxiv.org/abs/2210.05050).

**Neuro-symbolic programming** is a paradigm for artificial intelligence and cognitive computing that combines the strengths of both deep neural networks and symbolic reasoning.

**Deep neural networks** are a type of machine learning algorithms that are inspired by the structure and function of biological neural networks. They are particularly good at tasks such as image recognition, natural language processing, and decision making. However, they are not as good at tasks that require explicit reasoning, such as planning, problem solving, and understanding causal relationships.

**Symbolic reasoning**, on the other hand, is a type of reasoning that uses formal languages and logical rules to represent knowledge and perform tasks such as planning, problem solving, and understanding causal relationships. Symbolic reasoning systems are good at tasks that require explicit reasoning but are not as good at tasks that require pattern recognition or generalization, such as image recognition or natural language processing.

**Neuro-symbolic programming** aims to combine the strengths of both neural networks and symbolic reasoning to create AI systems that can perform a wide range of tasks. One way this is done is by using neural networks to extract information from data and then using symbolic reasoning to make inferences and decisions based on that information. Another way is by using symbolic reasoning to guide the generative process of neural networks and make them more interpretable.

**On a grander scale of things**, we believe that future computation platforms, such as wearables, SmartPhones, tables or notebooks will contain their own embedded LLMs (similar to GPT-3, ChatGPT, OPT or Bloom). 

<img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/assets/images/img1.png" width="720px">

These LLMs will be able to perform a wide range of computations, such as natural language understanding or decision making. Furthermore, neuro-symbolic computation engines will be able to learn concepts how to tackle unseen tasks and solve complex problems by querying various data sources for solutions and operating logical statements on top. 
In this turn, to ensure the generated content is in alignment with our goals, we need to develop ways to instruct, steer and control their generative processes. Therefore, our approach is an attempt to enable active and transparent flow control of these generative processes.

<img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/assets/images/img7.png" width="720px">

As shown in the figure above, we can think of it as shifting a probability mass from an input stream towards an output stream, in a contextualize manner. With proper designed conditions and expressions, we can also validate and steer the behavior towards our desired outcome, or repeat expressions that failed to fulfil our requirements. Our approach is to define a set of `fuzzy` operations that manipulate our data stream and conditions the LLMs. In essence, we consider all objects as symbols, and we can create operations that manipulate these symbols and generate new symbols. Each symbol can be interpreted as a statement. Multiple statements can be combined to form a logical expression.

Therefore, by chaining statements together we can build causal relationships and computations, instead of relying only on inductive approaches. Consequently, the outlook towards an updated computational stack resembles a neuro-symbolic computation engine at its core and, in combination with established frameworks, enables new applications. 


## 😶‍🌫️ How does it work?

We now show how we define our `Semantic API`, which is based on object-oriented and compositional design patterns. The `Symbol` class is the base class for all functional operations, which we refer also as a terminal symbol in the context of symbolic programming (fully resolved expressions). The Symbol class holds helpful operations and functions that can be interpreted as expressions to manipulate its content and evaluate to new Symbols. 

### 📚 Symbolic operations

Let us now define a Symbol and perform some basic manipulations. We start with a translation operation:

```python
sym = bd.Symbol("Welcome to our tutorial.")
sym.translate('German')
```
```bash
:[Output]: 
<class 'botdyn.expressions.Symbol'>(value=Willkommen zu unserem Tutorial.)
```

### Ranking objects

Our API can also perform basic data-agnostic operations to `filter`, `rank` or `extract` patterns. For example, we can rank a list of numbers:

```python
sym = Symbol(numpy.array([1, 2, 3, 4, 5, 6, 7]))
res = sym.rank(measure='numerical', order='descending')
```
```bash
:[Output]: 
<class 'botdyn.expressions.Symbol'>(value=['7', '6', '5', '4', '3', '2', '1'])
```

### Evaluating Expressions by best effort

As an inspiration, we relate to an approach demonstrated with [word2vec](https://arxiv.org/abs/1301.3781). 

**Word2Vec** generates dense vector representations of words by training a shallow neural network to predict a word given its neighbors in a text corpus. The resulting vectors are then used in a wide range of natural language processing applications, such as sentiment analysis, text classification, and clustering.

Below we can see an example how one can perform operations on the word embeddings (colored boxes).
The words are tokenized and mapped to a vector space, where we can perform semantic operations via vector arithmetics. 

<img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/assets/images/img3.png" width="450px">

Similar to word2vec we intend to preform contextualized operations on different symbols, however, instead of operating in the vector space, we operate in the natural language space. This gives us the ability to perform arithmetics on words, sentences, paragraphs, etc. and verify the results in a human readable format. 

The following examples shows how to evaluate such an expression via a string representation:

```python
Symbol('King - Man + Women').expression()
```
```bash
:[Output]:
<class 'botdyn.expressions.Symbol'>(value=Queen)
```

### Dynamic casting

We can also subtract sentences from each other, where our operations condition the neural computation engine to evaluate the Symbols by best effort. In the following example, it determines that the word `enemy` is present in the sentence, therefore, deletes it and replaces it with the word `friend` (which is added):

```python
res = bd.Symbol('Hello my enemy') - 'enemy' + 'friend'
```
```bash
:[Output]: 
<class 'botdyn.expressions.Symbol'>(value=Hello my friend)
```

What we also see is that the API performs dynamic casting, when data types are combined with a Symbol object. If an overloaded operation of the Symbol class is used, the Symbol class automatically casts the second object to Symbol. This is a convenient modality to perform operations between Symbol and other types of data, such as strings, integers, floats, lists, etc. without bloating the syntax.

### Fuzzy Comparisons

In this example we are fuzzily comparing two number objects, where the Symbol variant is only an approximation of `numpy.pi`. Given the context of the fuzzy equals `==` operation this comparison still succeeds and return `True`. 

```python
sym = bd.Symbol('3.1415...')
sym == numpy.pi
```
```bash
:[Output]:
True
```

### 🧠 Causal Reasoning

Our framework was built with the intention to enable reasoning capabilities on top of statistical inference of LLMs. Therefore, we can also perform deductive reasoning operations with our Symbol objects. For example, we can define a set of operations with rules that define the causal relationship between two symbols. The following example shows how the `&` is used to compute the logical implication of two symbols. 

```python
res = Symbol('The horn only sounds on Sundays.') & Symbol('I hear the horn.')
```
```bash
:[Output]:
<class 'botdyn.expressions.Symbol'>(value=It is Sunday.)
```

The current `&`-operation uses very simple logical statements `and`, `or` and `xor` via prompts to compute the logical implication. However, one can also define custom operations to perform more complex and robust logical operations, which also can use verification constraints to validate the outcomes and ensure a desired behavior, which we will explore in the next sections.

## Custom Operations

One can also define customized operations. For example, let us define a custom operation to generate a random integer between 0 and 10:

```python
class Demo(bd.Symbol):
    def __init__(self, value = '') -> None:
        super().__init__(value)
    
    @bd.zero_shot(prompt="Generate a random integer between 0 and 10.",
                  constraints=[
                      lambda x: x >= 0,
                      lambda x: x <= 10
                  ])
    def get_random_int(self) -> int:
        pass
```

As we show, the Semantic API uses Python `Decorators` to define operations. The `@bd.zero_shot` decorator is used to define a custom operation that does not require any demonstration examples, since the prompt is expressive enough. In the shown example, the `zero_shot` decorator takes in two arguments: `prompt` and `constraints`. The `prompt` argument is used to define the prompt that is used to condition on our desired  operation behavior. The `constraints` argument is used to define validation constraints of the computed outcome, and if it fulfils our expectations.

If the constraint is not fulfilled, the above implementation would reach out to the specified `default` implementation or default value. If no default implementation or value was found, the Semantic API would raise an `ConstraintViolationException`.

We also see that in the above example the return type is defined as `int`. Therefore, the resulting value from the wrapped function will be of type int. This works because our implementation uses auto-casting to a user specified return data type. If the cast fails, the Semantic API will raise a `ValueError`. If no return type is specified, the return type will be `Any`.

### Few-shot operations

The `@bd.few_shot` decorator is the more general version of `@bd.zero_shot` and is used to define a custom operation that requires demonstration examples. To give a more complete picture, we present the function signature of the `few_shot` decorator:

```python
def few_shot(prompt: str,
             examples: List[str], 
             constraints: List[Callable] = [],
             default: Optional[object] = None, 
             limit: int = 1,
             pre_processor: Optional[List[PreProcessor]] = None,
             post_processor: Optional[List[PostProcessor]] = None,
             **wrp_kwargs):
```

The `prompt` and `constraints` behave similar to the `zero_shot` decorator. However, as we see the `examples` and `limit` arguments are new. The `examples` argument is used to define a list of demonstration examples that are used to train the neural computation engine. The `limit` argument is used to define the maximum number of examples that are returned, give that there are more results. The `pre_processor` argument takes a list of `PreProcessor` objects which can be used to pre-process the input before it is fed into the neural computation engine. The `post_processor` argument takes a list of `PostProcessor` objects which can be used to post-process the output before it is returned to the user. The `wrp_kwargs` argument is used to pass additional arguments to the wrapped function, which are also stream-lined towards the neural computation engine and other engines.

Here is an example how to write an Japanese name generator:

```python
import botdyn as bd
class Demo(bd.Symbol):
    @bd.few_shot(prompt="Generate Japanese names: ",
                 examples=["愛子", "和花", "一郎", "和枝"],
                 limit=2,
                 constraints=[lambda x: len(x) > 1])
    def generate_japanese_names(self) -> list:
        return ['愛子', '和花'] # dummy implementation
```

Should the neural computation engine not be able to compute the desired outcome, it will reach out to the `default` implementation or default value. If no default implementation or value was found, the Semantic API would raise an exception.


## Prompt Design

The way all the above operations are performed is by using a `Prompt` class. The Prompt class is a container for all the information that is needed to define a specific operation. The Prompt class is also the base class for all other Prompt classes. 

Here is an example how to define a Prompt to enforce the neural computation engine to compare two values:

```python
class CompareValues(bd.Prompt):
    def __init__(self) -> bd.Prompt:
        super().__init__([
            "4 > 88 =>False",
            "-inf < 0 =>True",
            "inf > 0 =>True",
            "4 > 3 =>True",
            "1 < 'four' =>True",
            ...
        ])
```

For example, when calling the `<=` operation on two Symbols, the neural computation engine will evaluate the two Symbols and compare them based on the `CompareValues` prompt.

```python
res = bd.Symbol(1) <= bd.Symbol('one')
```

This statement evaluates to `True`, since the fuzzy compare operation was enforced to compare the two Symbols based on its semantic meaning, hence, this is the reason we call our framework also `Semantic API`.

```bash
:[Output]:
True
```

In a more general notion, depending on the context, hierarchy of the expression class and used operations the semantics of the Symbol manipulations may vary. To better illustrate this, we show our conceptual prompt design in the following figure:

<img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/assets/images/img4.png" width="350px">

The figure above shows the our context-based prompt design as a container of all the information that is provided to the neural computation engine to define a specific operation. 

Conceptually we consider three main prompt concepts: `Global Context`, `Operation` and `Examples`. The prompts can be curated either by inheritance or by composition. For example, the `Global Context` can be defined by inheriting from the `Expression` class and overriding the `static_context` property. An `Operation` and `Template` prompt can be created by providing an `PreProcessor` to modify the input. 

The `Global Context` concept is considered optional and can be defined in a static manner either by sub-classing the Expression class and overriding the `static_context` property, or at runtime by updating the `dynamic_context` kwargs. Nevertheless, the global context is conceptually meant to define the overall context of an expression. For example, if we want to operate in the context of a domain-specific language, without having to override each base class function over and over again.

The `Operation` prompt concept defines the behavior of an atomic operation and is therefore mandatory to express the nature of such an operation. For example, the `+` operation is used to add two Symbols together and therefore the `+`-operation prompt explains its behavior. 

`Examples` is another optional property and provides the neural computation engine with a set of demonstrations that are used to properly condition the engine. For example, the `+`-operation prompt can be conditioned on how to add numbers by providing a set of demonstrations, such as `1 + 1 = 2`, `2 + 2 = 4`, etc.

The `Template` prompt concept is optional and encapsulates the resulting prediction to enforce a specific format. For example, to generate HTML tags one can use a curated `<html>{{placeholder}}</html>` template. This template will enforce the neural computation engine to generate only HTML tags to replace the `{{placeholder}}` tag.


## 😑 Expressions

An `Expression` is a non-terminal symbol, which can be further evaluated. It inherits all the properties from Symbol and overrides the `__call__` function to evaluate its expressions or values. From the `Expression` class, all other expressions are derived. The Expression class also adds additional capabilities i.e. to `fetch` data from URLs, `search` on the internet or `open` files.

BotDynamics' API closely follows best practices and ideas from `PyTorch`, therefore, one can build complex expressions by combining multiple expressions as an computation graph. Each Expression has its own `forward` function, which has to be overridden. The `forward` function is used to define the behavior of the expression. The `forward` function is called by the `__call__` function, which is inherited from the Symbol class. The `__call__` function is used to evaluate the expression and return the result. The `__call__` function is also used to evaluate the expression in a lazy manner, which means that the expression is only evaluated when the result is needed. This is a very important feature, since it allows us to build complex expressions without having to evaluate the whole expression at once. We already implemented many useful expressions, which can be imported from the `botdyn.components` file.

Other important properties that are inherited from the Symbol class are the `_sym_return_type` and `static_context`. These two properties define the context in which the current Expression operations as described in the [Prompt Design](#prompt-design) section. The static_context therefore influences all operations held by the current Expression. The _sym_return_type ensures that after each Expression evaluation we obtain the desired return object type. This is usually the current type, but can be modified to return a different type. 

Expressions can of course have more complex structures and be further sub-classed, such as shown in the example of the `Sequence` expression in the following figure:

<img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/assets/images/img2.png" width="720px">

A Sequence expression can hold multiple expressions, which are evaluated at runtime.

### Sequence expressions

Here is an example how to define a Sequence expression:

```python
# first import all expressions
from botdyn import *
# define a sequence of expressions
Sequence(
    Clean(),
    Translate(),
    Outline(),
    Compose('Compose news:'),
)
```

### Stream expressions

As we saw earlier, we can create contextual prompts to define the context and operations of our neural engine. However, this also takes away a lot of the available context size and since i.e. the GPT-3 Davinci context length is already limited to 4097 tokens, this might quickly become a problem. Luckily, we can use the `Stream` processing expression. This expression opens up a data stream and performs chunks-based operations on the input stream. 

A Stream expression can easily be wrapped around other expressions. For example, the chunks can be processed with a `Sequence` expression, that allows multiple chained operations in sequential manner. Here is an example how to define such a Stream expression:

```python
Stream(Sequence(
    Clean(),
    Translate(),
    Outline(),
    Embed()
))
```
The stream operation chunks the long input text into smaller chunks and passes them to the inner expression, which returns us a `generator` object. In this case, to perform more complex operations, we open a stream and pass in a `Sequence` object which cleans, translates, outlines and embeds the information. 

The issue with that approach is only that the resulting chunks are processed independently from each other. This means that the context of the chunks is not preserved. To solve this issue, we can use the `Cluster` expression instead, where the independent chunks are recombined based on their similarity. We illustrate this in the following figure:

<img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/assets/images/img6.png" width="720px">

In the shown example we recombine all individual chunks again by clustering the information among the chunks. This gives us a way to consolidate contextually related information and recombine them in a meaningful way. Furthermore, the clustered information can then be labeled by looking / streaming through the values within the cluster and collecting the most relevant labels.

The full example is shown below:

```python
stream = Stream(Sequence(
    Clean(),
    Translate(),
    Outline(),
))
sym = Symbol('<some long text>')
res = Symbol([s for s in stream(sym)])
expr = Cluster()
expr(res)
```

In a next step, conceptually we could recursively repeat this process on each summary node, therefore, build a hierarchical clustering structure. Since each Node resembles the a summarized sub-set of the original information we can use it as an index of our larger content. The resulting tree can then be used to navigate and retrieve the original information, turning our large data stream problem into a search problem.

For searching in a vector space we can use dedicated libraries such as [Annoy](https://github.com/spotify/annoy), [Faiss](https://github.com/facebookresearch/faiss) or [Milvus](https://github.com/milvus-io/milvus). 

## ❌ Error Handling

A key idea of the BotDynamics API is to be able to also generate code. This also means that errors may occur that we need to handle in a contextual manner. In future terms we even want our API to self extend and therefore we need to be able to resolve issues automatically. To do so, we propose the `Try` expression, which has a fallback statement built in and retries an execution with dedicated error analysis and correction. This expression analyses the input and the error, and conditions itself to resolve the error by manipulating the original code. If the fallback expression succeeds, the result is returned, otherwise, this process is repeated for the number of `retries` specified. If the maximum number of retries is reached and the problem not resolved, the error is raised again. 

Let us assume, we have some executable code that was previously generated. However, by the nature of generative processes syntax errors may occur. And we also have the `Execute` expression, which takes in a symbol and tries to execute it. Naturally, this will fail and in the following example we illustrate how the `Try` expression resolves this syntactic error.

```python
expr = Try(expr=Execute())
sym = Symbol('a = int("3,")') # some code with a syntax error
res = expr(sym)
```

The resulting output is the evaluated code, which was corrected:

```bash
:Output:
a = 3
```

Nevertheless, we are aware that not all errors are as simple as the shown syntactic error example, which can be resolved automatically. Many errors occur due to semantic misconceptions. Such issues, require contextual information and therefore we are further exploring means towards more sophisticated error handling mechanism.
This includes also the usage of streams and clustering to resolve errors in a more hierarchical contextual manner.


## 🕷️ Explainability, Testing & Debugging

Perhaps one of the greatest benefits when using neuro-symbolic programming is that we can get a clear understanding of how well our LLMs understand atomic operations, if they fail and more specifically, when they fail to be able to follow their StackTraces to determine the failure points. Neuro-symbolic programming allows us to debug the model predictions and understand how they came about. Furthermore, we can Unit Test them to detect conceptual misalignments. 

### Unit Testing Models

Since our premise is to divide and conquer complex problems, we can now curate conceptual Unit Test and target very specific and trackable sub-problems. The resulting measure, i.e. success rate of model predictions, can then be used to evaluate their conceptual performance, and hint towards undesired flaws or biases.

This allows us now to design domain-specific benchmarks and see how well general learners, such as GPT-3, adapt to these tasks. 

For example, we can write a fuzzy comparison operation, that can take in digits and strings alike and perform a semantic comparison. LLMs can then be asked to evaluate these expressions. Often times, these LLMs still fail to understand the semantic meaning these tokens and predict wrong answers. 

The following code snipped shows a Unit Test to perform semantic comparison of numbers (between digits and strings):

```python
import unittest
from botdyn import *

class TestComposition(unittest.TestCase):
  def test_compare(self):
      res = Symbol(10) > Symbol(5)
      self.assertTrue(res)
      res = Symbol(1) < Symbol('five')
      self.assertTrue(res)
      ...
```

### 🔥Debugging

When creating very complex expressions, we we debug them by using the `Trace` expression, which allows us to print out and follow the StackTrace of the neuro-symbolic operations. Combined with the `Log` expression, which creates a dump of all prompts and results to a log file, we can analyze where our models potentially failed.


### Examples

In the following example we create a news summary expression that crawls the given URL and streams the site content through multiple expressions. The outcome is a news website that is created based on the crawled content. The `Trace` expression allows us to follow the StackTrace of the operations and see what operations are currently executed. If we open the `outputs/engine.log` file we can see the dumped traces with all the prompts and results.

```python
# crawling the website and creating an own website based on its facts
news = News(url='https://www.cnbc.com/cybersecurity/',
            pattern='cnbc',
            filters=ExcludeFilter('sentences about subscriptions, licensing, newsletter'),
            render=True)
expr = Log(Trace(news))
res = expr()
```

Here is the corresponding StackTrace of the model:

<img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/assets/images/img8.png" width="900px">

The above code creates a webpage with the crawled content from the original source. See the preview below, the entire [rendered webpage image here](https://raw.githubusercontent.com/Xpitfire/botdynamics/main/examples/results/news.png) and resulting [code of webpage here](https://raw.githubusercontent.com/Xpitfire/botdynamics/main/examples/results/news.html). 


<img src="https://raw.githubusercontent.com/Xpitfire/botdynamics/main/examples/results/news_prev.png" width="900px">

Launch and explore the notebook here:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Xpitfire/botdynamics/HEAD)

There many more examples in the [examples folder](examples/) and in the [notebooks folder](notebooks/). You can also explore the test cases in the [tests folder](tests/).


## 📈 Concepts for Data Collection \& Analytics

BotDynamics is by design a data-driven framework. This means that we can collect data from API interactions while we provide the requested responses. For very agile, dynamic adaptations or prototyping we can integrate user desired behavior quickly into existing prompts. However, we can also log the user queries and model predictions to make them available for post-analysis, and fine-tuning. Therefore, we can quickly iterate, customize and improve our model responses based on real-world data.

In the following example, we show how we can use an `Output` expression to pass a handler function and access input prompts of the model and model predictions. These, can be used for data collection and later fine-tunning stages. The handler function provides a dictionary and offers keys for `input` and `output` values. The content can then be sent to a data pipeline for further processing.

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

Since we called verbose, we can also see the console print of the `Output` expression:

```bash
Input: (['Translate the following text into German:\n\nHello World!'],)
Expression: <bound method Symbol.translate of <class 'botdyn.symbol.Symbol'>(value=Hello World!)>
args: ('German',) kwargs: {'input_handler': <function OutputEngine.forward.<locals>.input_handler at ...
Dictionary: {'wrp_self': <class 'botdyn.components.Output'>(value=None), 'func': <function Symbol.output.<locals>._func at ...
Output: Hallo Welt!
```


## 🤖 Engines

To to limited compute resources we currently only rely on OpenAI's API. However, given the right compute resources, we could use local machines to avoid high latencies and costs, with alternative engines such as OPT or Bloom. This would allow for recursive executions, loops, and more complex expressions.

However, we already integrated a set of useful engines that are capable of providing language tokens to perform symbolic operations. 

### Speech Engine

To perform speech transcription we use `whisper`. The following example shows how to transcribe an audio file and return the text:

```python
expr = Expression()
res = expr.speech('examples/audio.mp3')
```

```bash
:Output:
I may have overslept.
```

### OCR Engine

To perform OCR we use `APILayer`. The following example shows how to transcribe an image and return the text:

```python
expr = Expression()
res = expr.ocr('https://media-cdn.tripadvisor.com/media/photo-p/0f/da/22/3a/rechnung.jpg')
```

The OCR engine returns a dictionary with a key `all_text` where the full text is stored. See more details in their documentation [here](https://apilayer.com/marketplace/image_to_text-api).

```bash
:Output:
China Restaurant\nMaixim,s\nSegeberger Chaussee 273\n22851 Norderstedt\nTelefon 040/529 16 2 ...
```


### Search Engine

To perform search queries we use `SerpApi` with `Google` backend. The following example shows how to search for a query and return the results:

```python
expr = Expression()
res = expr.search('Birthday of Barack Obama')
```

```bash
:Output:
August 4, 1961
```

### WebCrawler Engine

To perform web crawling we use `Selenium`. The following example shows how to crawl a website and return the results:

```python
expr = Expression()
res = expr.fetch(url="https://www.google.com/", 
                 pattern="google")
```
The `pattern` property can be used to detect if the document as been loaded correctly. If the pattern is not found, the crawler will timeout and return an empty result.

```bash
:Output:
GoogleKlicke hier, wenn du nach einigen Sekunden nicht automatisch weitergeleitet wirst.GmailBilderAnmelden ...
```

### Drawing Engine

To render nice images from text description we use `DALL·E 2`. The following example shows how to draw a text description and return the image:

```python
expr = Expression('a cat with a hat')
res = expr.draw()
```

```bash
:Output:
https://oaidalleapiprodscus.blob.core.windows.net/private/org-l6FsXDfth6Uct ...
```

Don't worry, we would never hide an image of a cat with a hat from you. Here is the image preview and [link](https://oaidalleapiprodscus.blob.core.windows.net/private/org-l6FsXDfth6UctywtAPNtbH6K/user-vrlXYI3y3uHLeW7OkYJd7K2c/img-u0R3r9KQQ0soqnx0wLsa3ShS.png?st=2023-01-13T19%3A16%3A00Z&se=2023-01-13T21%3A16%3A00Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-13T18%3A19%3A36Z&ske=2023-01-14T18%3A19%3A36Z&sks=b&skv=2021-08-06&sig=Fl13UoQiLdjnBqnYG6ttnnEVfqbGTiuYk/pgpap8V%2BY%3D):

<img src="https://oaidalleapiprodscus.blob.core.windows.net/private/org-l6FsXDfth6UctywtAPNtbH6K/user-vrlXYI3y3uHLeW7OkYJd7K2c/img-u0R3r9KQQ0soqnx0wLsa3ShS.png?st=2023-01-13T19%3A16%3A00Z&se=2023-01-13T21%3A16%3A00Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-13T18%3A19%3A36Z&ske=2023-01-14T18%3A19%3A36Z&sks=b&skv=2021-08-06&sig=Fl13UoQiLdjnBqnYG6ttnnEVfqbGTiuYk/pgpap8V%2BY%3D" width="200px">


### File Engine

To perform file operations we currently only use the file system of your OS. We support currently only PDF files and plain text files. This is a very early stage and we are working on a more sophisticated file system access and remote storages. The following example shows how to read a PDF file and return the text:

```python
expr = Expression()
res = expr.open('./LICENSE')
```

```bash
:Output:
MIT License\n\nCopyright (c) 2023 Marius-Constantin Dinu\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation ...
```


### CLIP Engine

To perform text-based image few-shot classification we use `CLIP`. This implementation is very experimental and does not conceptually fully integrate the way we intend it, since the embeddings of CLIP and GPT-3 do not yet aline. This is an open problem for future research. The following example shows how to classify the image of our generate cat from above and return the results as an array of probabilities:

```python
expr = Expression()
res = expr.vision('https://oaidalleapiprodscus.blob.core.windows.net/private/org-l6FsXDfth6...', 
                  ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'])
```

```bash
:Output:
array([[9.72840726e-01, 6.34790864e-03, 2.59368378e-03, 3.41371237e-03,
        3.71197984e-03, 8.53193272e-03, 1.03346225e-04, 2.08464009e-03,
        1.77942711e-04, 1.94185617e-04]], dtype=float32)
```


## ⚡Limitations

Uff... this is a long list. We are still in the early stages of development and are working hard to overcome these limitations. Just to name a few:

Engineering challenges:
* Our framework is still in its early stages of development and is not yet meant for production use. For example, the Stream class only estimates the prompt size by an approximation, which can be easily exceeded. One can also create more sophisticated prompt hierarchies and dynamically adjust the global context based on a state-based approach. This would allow to make consistent predictions even for long text streams.
* The code may not be complete and is not yet optimized for speed and memory usage, and uses API-based LLMs due to limitations of compute resources.
* Code coverage is not yet complete and we are still working on the documentation.
* Integrate with a more diverse set of models from [HuggingFace](https://huggingface.co/) or other platforms.
* Currently we did not account for multi-threading and multi-processing.
* Vector-based search indexes, i.e. [Annoy](https://github.com/spotify/annoy), [Faiss](https://github.com/facebookresearch/faiss) or [Milvus](https://github.com/milvus-io/milvus), are not yet integrated into the framework to enable fast content retrieval.

Research challenges:
* The experimental integration of CLIP is meant to align image and text embeddings. To enable decision-making of LLMs based on observations and perform symbolic operations on objects in images or videos would be a huge leap forward. This would perfectly integrate with reinforcement learning approaches and enable us to control policies in a systematic way (see also [GATO](https://www.deepmind.com/publications/a-generalist-agent)). Therefore, we need to train large multi-modal variants with image / video data and text data, describing in high details the scenes to obtain neuro-symbolic computation engines that can perform semantic operations similar to `move-towards-tree`, `open-door`, etc.
* Generalist LLMs are still highly over-parameterized and hardware has not yet caught up to host these models on arbitrary day-to-day machines. This limits the applicability of our approach not only on small data streams, but also gives high latencies and therefore limits the amount of complexity and expressiveness we can achieve with our expressions.


## 🥠 Future Work

We are constantly working on improving the framework and are open to any suggestions. However, we try to think ahead of time and have some general ideas for future work in mind:

* Meta-Learning Semantic Concepts on top of Neuro-Symbolic Expressions
* Self-evolving and self-healing API

We believe that LLMs as neuro-symbolic computation engines enable us a new class of applications, with tools and API that can self-analyze and self-heal. We are excited to see what the future brings and are looking forward to your feedback and contributions.

## Conclusion

We have presented 



## 👥 References, Related Work \& Credits

This project is inspired by the following works, but not limited to them:

* [The Algebraic Theory of Context-Free Languages](http://www-igm.univ-mlv.fr/~berstel/Mps/Travaux/A/1963-7ChomskyAlgebraic.pdf)
* [Tracr: Compiled Transformers as a Laboratory for Interpretability](https://arxiv.org/abs/2301.05062)
* [How can computers get common sense?](https://www.science.org/doi/10.1126/science.217.4566.1237)
* [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)
* [Neuro-symbolic programming](https://arxiv.org/abs/2210.05050)
* [Fuzzy Sets](https://web.archive.org/web/20150813153834/http://www.cs.berkeley.edu/~zadeh/papers/Fuzzy%20Sets-Information%20and%20Control-1965.pdf)
* [An early approach toward graded identity and graded membership in set theory](https://www.sciencedirect.com/science/article/abs/pii/S0165011409005326?via%3Dihub)
* [From Statistical to Causal Learning](https://arxiv.org/abs/2204.00607)
* [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
* [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
* [Aligning Language Models to Follow Instructions](https://openai.com/blog/instruction-following/)
* [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
* [Measuring and Narrowing the Compositionality Gap in Language Models](https://ofir.io/self-ask.pdf)
* [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)
* [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586)
* [Understanding Stereotypes in Language Models: Towards Robust Measurement and Zero-Shot Debiasing](https://arxiv.org/abs/2212.10678)
* [Connectionism and Cognitive Architecture: A Critical Analysis](https://ruccs.rutgers.edu/images/personal-zenon-pylyshyn/proseminars/Proseminar13/ConnectionistArchitecture.pdf)
* [Unit Testing for Concepts in Neural Networks](https://arxiv.org/abs/2208.10244)
* [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)


### Acknowledgements

Also this is a long list. Great thanks to my colleagues and friends at the [Institute for Machine Learning at Johannes Kepler University (JKU), Linz](https://www.jku.at/institut-fuer-machine-learning/) for their great support and feedback. Also special thanks to [Dynatrace Research](https://engineering.dynatrace.com/research/) for supporting this project. Great thanks to the [OpenAI](https://openai.com/), [GitHub](https://github.com/) and [Microsoft Research](https://www.microsoft.com/en-us/research/) teams for their great work and making their API and tools available to the public, and thanks to all the people who contributed to this project. Be it by providing feedback, bug reports, code, or just by using the framework. We are very grateful for your support. 


### Contribution

If you want to contribute to this project, please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on our code of conduct, and the process for submitting pull requests to us. Any contributions are highly appreciated.

### 📜 Citation

```bibtex
@software{Dinu_BotDynamics_2022,
  author = {Dinu, Marius-Constantin and Co, Pilot and GPT-3, Davinci-003},
  title = {{BotDynamics}},
  url = {https://github.com/Xpitfire/botdynamics},
  version = {0.0.1},
  month = {11},
  year = {2022}
}
```

### 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Like this project?

If you like this project, leave a star ⭐️ and share it with your friends and colleagues.
And if you want to support this project even further, please consider donating to support the continuous development of this project. Thank you!

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=WCWP5D2QWZXFQ)

We are also looking for contributors or investors to grow and support this project. If you are interested, please contact us.

### 📫 Contact

If you have any questions about this project, please contact us via [email](mailto:office@alphacoreai.eu), on our [website](www.alphacoreai.eu/) or find us on Discord:

[![Discord](https://img.shields.io/discord/768087161878085643?label=Discord&logo=Discord&logoColor=white)](https://discord.gg/azDQxCHeDA)

If you want to contact me directly, you can reach me directly on [LinkedIn](https://www.linkedin.com/in/mariusconstantindinu/), on [Twitter](https://twitter.com/DinuMariusC) at my personal [website](https://www.dinu.at/).
