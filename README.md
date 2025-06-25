# **SymbolicAI: A neuro-symbolic perspective on LLMs**
<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/refs/heads/main/assets/images/banner.png">

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge)](https://extensityai.gitbook.io/symbolicai)
[![Arxiv](https://img.shields.io/badge/Paper-32758e?style=for-the-badge)](https://arxiv.org/abs/2402.00854)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-yellow?style=for-the-badge)](https://deepwiki.com/ExtensityAI/symbolicai)

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/dinumariusc.svg?style=social&label=@DinuMariusC)](https://twitter.com/DinuMariusC) [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/symbolicapi.svg?style=social&label=@ExtensityAI)](https://twitter.com/ExtensityAI)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/futurisold.svg?style=social&label=@futurisold)](https://x.com/futurisold)

</div>

---

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/preview.gif">

## What is SymbolicAI?

SymbolicAI is a **neuro-symbolic** framework, combining classical Python programming with the differentiable, programmable nature of LLMs. In this README, we'll introduce two key concepts that define SymbolicAI: **primitives** and **contracts**.

### Primitives
At the core of SymbolicAI are `Symbol` objects—each one comes with a set of tiny, composable operations that feel like native Python. Think of them as your building blocks for semantic reasoning. Right now, we support a wide range of primitives:

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

If you want to see how these primitives are used in practice, check out the docs [here](https://extensityai.gitbook.io/symbolicai/features/primitives).

### Contracts

They say LLMs hallucinate—but your code can't afford to. That's exactly why SymbolicAI brings **Design by Contract** principles into the world of LLMs. Instead of relying solely on post-hoc testing, contracts help build correctness directly into your design. All you need is some data models and a decorator. Read more about contracts [here](https://deepwiki.com/ExtensityAI/symbolicai/7.1-contract-validation-system) and [here](https://extensityai.gitbook.io/symbolicai/features/contracts).

## Installation

### Core Features

To get started with SymbolicAI, you can install it using pip:

```bash
pip install symbolicai
```

#### Setting up a neurosymbolic API Key

Before using SymbolicAI, you need to set up API keys for the various engines. Currently, SymbolicAI supports the following neurosymbolic engines through API: OpenAI, Anthropic. We also support {doc}`local neurosymbolic engines <ENGINES/local_engine>`, such as llama.cpp and huggingface.

```bash
# Linux / MacOS
export NEUROSYMBOLIC_ENGINE_API_KEY="…"
export NEUROSYMBOLIC_ENGINE_MODEL="…"
```

```bash
# Windows (PowerShell)
$Env:NEUROSYMBOLIC_ENGINE_API_KEY="…"
$Env:NEUROSYMBOLIC_ENGINE_MODEL="…"
```

```bash
# Jupyter Notebooks
%env NEUROSYMBOLIC_ENGINE_API_KEY=…
%env NEUROSYMBOLIC_ENGINE_MODEL=…
```

#### Optional Features

SymbolicAI uses multiple engines to process text, speech and images. We also include search engine access to retrieve information from the web. To use all of them, you will need to also install the following dependencies and assign the API keys to the respective engines.

```bash
pip install "symbolicai[wolframalpha]"
pip install "symbolicai[whisper]"
pip install "symbolicai[selenium]"
pip install "symbolicai[serpapi]"
pip install "symbolicai[pinecone]"
```

Or, install all optional dependencies at once:

```bash
pip install "symbolicai[all]"
```

And export the API keys, for example:

```bash
# Linux / MacOS
export SYMBOLIC_ENGINE_API_KEY="<WOLFRAMALPHA_API_KEY>"
export SEARCH_ENGINE_API_KEY="<SERP_API_KEY>"
export OCR_ENGINE_API_KEY="<APILAYER_API_KEY>"
export INDEXING_ENGINE_API_KEY="<PINECONE_API_KEY>"

# Windows (PowerShell)
$Env:SYMBOLIC_ENGINE_API_KEY="<WOLFRAMALPHA_API_KEY>"
$Env:SEARCH_ENGINE_API_KEY="<SERP_API_KEY>"
$Env:OCR_ENGINE_API_KEY="<APILAYER_API_KEY>"
$Env:INDEXING_ENGINE_API_KEY="<PINECONE_API_KEY>"
```

See below for the entire list of keys that can be set via environment variables or a configuration file.

#### Additional Requirements

**SpeechToText Engine**: Install `ffmpeg` for audio processing (based on OpenAI's [whisper](https://openai.com/blog/whisper/))

```bash
# Linux
sudo apt update && sudo apt install ffmpeg

# MacOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

**WebCrawler Engine**: For `selenium`, we automatically install the driver with `chromedriver-autoinstaller`. Currently we only support Chrome as the default browser.

## Configuration Management

SymbolicAI now features a configuration management system with priority-based loading. The configuration system looks for settings in three different locations, in order of priority:

1. **Debug Mode** (Current Working Directory)
   - Highest priority
   - Only applies to `symai.config.json`
   - Useful for development and testing

2. **Environment-Specific Config** (Python Environment)
   - Second priority
   - Located in `{python_env}/.symai/`
   - Ideal for project-specific settings

3. **Global Config** (Home Directory)
   - Lowest priority
   - Located in `~/.symai/`
   - Default fallback for all settings

### Configuration Files

The system manages three main configuration files:
- `symai.config.json`: Main SymbolicAI configuration
- `symsh.config.json`: Shell configuration
- `symserver.config.json`: Server configuration

### Viewing Your Configuration

Before using the package, we recommend inspecting your current configuration setup using the command below. This will create all the necessary configuration files.

```bash
symconfig
```

This command will show:
- All configuration locations
- Active configuration paths
- Current settings (with sensitive data truncated)

### Configuration Priority Example

```console
my_project/              # Debug mode (highest priority)
└── symai.config.json    # Only this file is checked in debug mode

{python_env}/.symai/     # Environment config (second priority)
├── symai.config.json
├── symsh.config.json
└── symserver.config.json

~/.symai/                # Global config (lowest priority)
├── symai.config.json
├── symsh.config.json
└── symserver.config.json
```

If a configuration file exists in multiple locations, the system will use the highest-priority version. If the environment-specific configuration is missing or invalid, the system will automatically fall back to the global configuration in the home directory.

### Best Practices

- Use the global config (`~/.symai/`) for your default settings
- Use environment-specific configs for project-specific settings
- Use debug mode (current directory) for development and testing
- Run `symconfig` to inspect your current configuration setup

This addition to the README clearly explains:
1. The priority-based configuration system
2. The different configuration locations and their purposes
3. How to view and manage configurations
4. Best practices for configuration management

### Configuration File

You can specify engine properties in a symai.config.json file in your project path. This will replace the environment variables. The default configuration file that will be created is:
```json
{
    "NEUROSYMBOLIC_ENGINE_API_KEY": "",
    "NEUROSYMBOLIC_ENGINE_MODEL": "",
    "SYMBOLIC_ENGINE_API_KEY": "",
    "SYMBOLIC_ENGINE": "",
    "EMBEDDING_ENGINE_API_KEY": "",
    "EMBEDDING_ENGINE_MODEL": "",
    "DRAWING_ENGINE_MODEL": "",
    "DRAWING_ENGINE_API_KEY": "",
    "SEARCH_ENGINE_API_KEY": "",
    "SEARCH_ENGINE_MODEL": "",
    "INDEXING_ENGINE_API_KEY": "",
    "INDEXING_ENGINE_ENVIRONMENT": "",
    "TEXT_TO_SPEECH_ENGINE_MODEL": "",
    "TEXT_TO_SPEECH_ENGINE_API_KEY": "",
    "SPEECH_TO_TEXT_ENGINE_MODEL": "",
    "VISION_ENGINE_MODEL": "",
    "OCR_ENGINE_API_KEY": "",
    "COLLECTION_URI": "",
    "COLLECTION_DB": "",
    "COLLECTION_STORAGE": "",
    "SUPPORT_COMMUNITY": false,
}
```
Example of a configuration file with all engines enabled:
```json
{
    "NEUROSYMBOLIC_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "NEUROSYMBOLIC_ENGINE_MODEL": "gpt-4o",
    "SYMBOLIC_ENGINE_API_KEY": "<WOLFRAMALPHA_API_KEY>",
    "SYMBOLIC_ENGINE": "wolframalpha",
    "EMBEDDING_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "EMBEDDING_ENGINE_MODEL": "text-embedding-3-small",
    "DRAWING_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "DRAWING_ENGINE_MODEL": "dall-e-3",
    "VISION_ENGINE_MODEL": "openai/clip-vit-base-patch32",
    "SEARCH_ENGINE_API_KEY": "<PERPLEXITY_API_KEY>",
    "SEARCH_ENGINE_MODEL": "llama-3.1-sonar-small-128k-online",
    "OCR_ENGINE_API_KEY": "<APILAYER_API_KEY>",
    "SPEECH_TO_TEXT_ENGINE_MODEL": "turbo",
    "TEXT_TO_SPEECH_ENGINE_MODEL": "tts-1",
    "INDEXING_ENGINE_API_KEY": "<PINECONE_API_KEY>",
    "INDEXING_ENGINE_ENVIRONMENT": "us-west1-gcp",
    "COLLECTION_DB": "ExtensityAI",
    "COLLECTION_STORAGE": "SymbolicAI",
    "SUPPORT_COMMUNITY": true
}
```

With these steps completed, you should be ready to start using SymbolicAI in your projects.

> **[NOTE]**: Our framework allows you to support us train models for local usage by enabling the data collection feature. On application startup we show the terms of services and you can activate or disable this community feature. We do not share or sell your data to 3rd parties and only use the data for research purposes and to improve your user experience. To change this setting open the `symai.config.json` and turn it on/off by setting the `SUPPORT_COMMUNITY` property to `True/False` via the config file or the respective environment variable.
> **[NOTE]**: By default, the user warnings are enabled. To disable them, export `SYMAI_WARNINGS=0` in your environment variables.

### Running tests
Some examples of running tests locally:
```bash
# Run all tests
pytest tests
# Run mandatory tests
pytest -m mandatory
```
Be sure to have your configuration set up correctly before running the tests. You can also run the tests with coverage to see how much of the code is covered by tests:
```bash
pytest --cov=symbolicai tests
```

## 🪜 Next Steps

Now, there are tools like DeepWiki that provide better documentation than we could ever write, and we don’t want to compete with that; we'll correct it where it's plain wrong. Please go read SymbolicAI's DeepWiki [page](https://deepwiki.com/ExtensityAI/symbolicai/). There's a lot of interesting stuff in there. Last but not least, check out our [paper](https://arxiv.org/abs/2402.00854) that describes the framework in detail. If you like watching videos, we have a series of tutorials that you can find [here](https://extensityai.gitbook.io/symbolicai/tutorials/video_tutorials).

## 📜 Citation

```bibtex
@software{Dinu_SymbolicAI_2022,
  author = {Dinu, Marius-Constantin},
  editor = {Leoveanu-Condrei, Claudiu},
  title = {{SymbolicAI: A Neuro-Symbolic Perspective on Large Language Models (LLMs)}},
  url = {https://github.com/ExtensityAI/symbolicai},
  month = {11},
  year = {2022}
}
```

## 📝 License

This project is licensed under the BSD-3-Clause License - refer to [the docs](https://symbolicai.readthedocs.io/en/latest/LICENSE.html).

## Like this Project?

If you appreciate this project, please leave a star ⭐️ and share it with friends and colleagues. To support the ongoing development of this project even further, consider donating. Thank you!

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg?style=for-the-badge)](https://www.paypal.com/donate/?hosted_button_id=WCWP5D2QWZXFQ)

We are also seeking contributors or investors to help grow and support this project. If you are interested, please reach out to us.

## 📫 Contact

Feel free to contact us with any questions about this project via [email](mailto:office@extensity.ai), through our [website](https://extensity.ai/), or find us on Discord:
[![Discord](https://img.shields.io/discord/768087161878085643?label=Discord&logo=Discord&logoColor=white?style=for-the-badge)](https://discord.gg/QYMNnh9ra8)
