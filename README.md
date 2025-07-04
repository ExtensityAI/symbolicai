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

SymbolicAI is a **neuro-symbolic** framework, combining classical Python programming with the differentiable, programmable nature of LLMs in a way that actually feels natural in Python.
It's built to not stand in the way of your ambitions.
It's easily extensible and customizable to your needs by virtue of its modular design.
It's quite easy to [write your own engine](https://extensityai.gitbook.io/symbolicai/engines/custom_engine), [host locally](https://extensityai.gitbook.io/symbolicai/engines/local_engine) an engine of your choice, or interface with tools like [web search](https://extensityai.gitbook.io/symbolicai/engines/search_engine) or [image generation](https://extensityai.gitbook.io/symbolicai/engines/drawing_engine).
To keep things concise in this README, we'll introduce two key concepts that define SymbolicAI: **primitives** and **contracts**.

 > ❗️**NOTE**❗️ The framework's name is intended to credit the foundational work of Allen Newell and Herbert Simon that inspired this project.

### Primitives
At the core of SymbolicAI are `Symbol` objects—each one comes with a set of tiny, composable operations that feel like native Python.
```python
from symai import Symbol
```

`Symbol` comes in **two flavours**:

1. **Syntactic** – behaves like a normal Python value (string, list, int ‐ whatever you passed in).
2. **Semantic**  – is wired to the neuro-symbolic engine and therefore *understands* meaning and
   context.

Why is syntactic the default?
Because Python operators (`==`, `~`, `&`, …) are overloaded in `symai`.
If we would immediately fire the engine for *every* bitshift or comparison, code would be slow and could produce surprising side-effects.
Starting syntactic keeps things safe and fast; you opt-in to semantics only where you need them.

#### How to switch to the semantic view

1. **At creation time**

   ```python
   S = Symbol("Cats are adorable", semantic=True) # already semantic
   print("feline" in S) # => True
   ```

2. **On demand with the `.sem` projection** – the twin `.syn` flips you back:

   ```python
   S = Symbol("Cats are adorable") # default = syntactic
   print("feline" in S.sem) # => True
   print("feline" in S)     # => False
   ```

3. Invoking **dot-notation operations**—such as `.map()` or any other semantic function—automatically switches the symbol to semantic mode:

   ```python
    S = Symbol(['apple', 'banana', 'cherry', 'cat', 'dog'])
    print(S.map('convert all fruits to vegetables'))
    # => ['carrot', 'broccoli', 'spinach', 'cat', 'dog']
   ```

Because the projections return the *same underlying object* with just a different behavioural coat, you can weave complex chains of syntactic and semantic operations on a single symbol. Think of them as your building blocks for semantic reasoning. Right now, we support a wide range of primitives; check out the docs [here](https://extensityai.gitbook.io/symbolicai/features/primitives), but here's a quick snack:

| Primitive/Operator | Category         | Syntactic | Semantic | Description |
|--------------------|-----------------|:---------:|:--------:|-------------|
| `==`               | Comparison      | ✓         | ✓        | Tests for equality. Syntactic: literal match. Semantic: fuzzy/conceptual equivalence (e.g. 'Hi' == 'Hello'). |
| `+`                | Arithmetic      | ✓         | ✓        | Syntactic: numeric/string/list addition. Semantic: meaningful composition, blending, or conceptual merge. |
| `&`                | Logical/Bitwise | ✓         | ✓        | Syntactic: bitwise/logical AND. Semantic: logical conjunction, inference, e.g., context merge. |
| `symbol[index] = value` | Iteration        | ✓         | ✓        | Set item or slice. |
| `.startswith(prefix)`    | String Helper    | ✓         | ✓        | Check if a string starts with given prefix (in both modes). |
| `.choice(cases, default)` | Pattern Matching|           | ✓        | Select best match from provided cases. |
| `.foreach(condition, apply)`| Execution Control |         | ✓        | Apply action to each element. |
| `.cluster(**clustering_kwargs?)`              | Data Clustering  |         | ✓        | Cluster data into groups semantically. (uses sklearn's DBSCAN)|
| `.similarity(other, metric?, normalize?)` | Embedding    |         | ✓        | Compute similarity between embeddings. |
| ... | ...    |   ...|  ...        | ... |

### Contracts

They say LLMs hallucinate—but your code can't afford to. That's why SymbolicAI brings **Design by Contract** principles into the world of LLMs. Instead of relying solely on post-hoc testing, contracts help build correctness directly into your design, everything packed into a decorator that will operate on your defined data models and validation constraints:
```python
from symai import Expression
from symai.strategy import contract
from symai.models import LLMDataModel # Compatible with Pydantic's BaseModel
from pydantic import Field, field_validator

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
#  Data models                                              ▬
#  – clear structure + rich Field descriptions power        ▬
#    validation, automatic prompt templating & remedies     ▬
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class DataModel(LLMDataModel):
    some_field: some_type = Field(description="very descriptive field", and_other_supported_options_here="...")

    @field_validator('some_field')
    def validate_some_field(cls, v):
        # Custom basic validation logic can be added here too besides pre/post
        valid_opts = ['A', 'B', 'C']
        if v not in valid_opts:
            raise ValueError(f'Must be one of {valid_opts}, got "{v}".')
        return v

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
#  The contracted expression class                          ▬
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@contract(
    # ── Remedies ─────────────────────────────────────────── #
    pre_remedy=True,        # Try to fix bad inputs automatically
    post_remedy=True,       # Try to fix bad LLM outputs automatically
    accumulate_errors=True, # Feed history of errors to each retry
    verbose=True,           # Nicely displays progress in terminal
    remedy_retry_params=dict(tries=3, delay=0.4, max_delay=4.0,
                             jitter=0.15, backoff=1.8, graceful=False),
)
class Agent(Expression):
    #
    # High-level behaviour:
    #  *. `prompt` – a *static* description of what the LLM must do (mandatory)
    #  1. `pre`    – sanity-check inputs (optional)
    #  2. `act`    – mutate state (optional)
    #  3. LLM      – generate expected answer (handled by SymbolicAI engine)
    #  4. `post`   – ensure answer meets semantic rules (optional)
    #  5. `forward` (mandatory)
    #     • if contract succeeded → return type validated LLM object
    #     • else                  → graceful fallback answer
    # ...
```

Because we don't want to bloat this README file with long Python snippets, learn more about contracts [here](https://deepwiki.com/ExtensityAI/symbolicai/7.1-contract-validation-system) and [here](https://extensityai.gitbook.io/symbolicai/features/contracts).

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

### Configuration File

You can specify engine properties in a `symai.config.json` file in your project path. This will replace the environment variables.
Example of a configuration file with all engines enabled:
```json
{
    "NEUROSYMBOLIC_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "NEUROSYMBOLIC_ENGINE_MODEL": "gpt-4o",
    "SYMBOLIC_ENGINE_API_KEY": "<WOLFRAMALPHA_API_KEY>",
    "SYMBOLIC_ENGINE": "wolframalpha",
    "EMBEDDING_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "EMBEDDING_ENGINE_MODEL": "text-embedding-3-small",
    "SEARCH_ENGINE_API_KEY": "<PERPLEXITY_API_KEY>",
    "SEARCH_ENGINE_MODEL": "sonar",
    "TEXT_TO_SPEECH_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "TEXT_TO_SPEECH_ENGINE_MODEL": "tts-1",
    "INDEXING_ENGINE_API_KEY": "<PINECONE_API_KEY>",
    "INDEXING_ENGINE_ENVIRONMENT": "us-west1-gcp",
    "DRAWING_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "DRAWING_ENGINE_MODEL": "dall-e-3",
    "VISION_ENGINE_MODEL": "openai/clip-vit-base-patch32",
    "OCR_ENGINE_API_KEY": "<APILAYER_API_KEY>",
    "SPEECH_TO_TEXT_ENGINE_MODEL": "turbo",
    "SUPPORT_COMMUNITY": true
}
```

With these steps completed, you should be ready to start using SymbolicAI in your projects.

> ❗️**NOTE**❗️Our framework allows you to support us train models for local usage by enabling the data collection feature. On application startup we show the terms of services and you can activate or disable this community feature. We do not share or sell your data to 3rd parties and only use the data for research purposes and to improve your user experience. To change this setting open the `symai.config.json` and turn it on/off by setting the `SUPPORT_COMMUNITY` property to `True/False` via the config file or the respective environment variable.

> ❗️**NOTE**❗️By default, the user warnings are enabled. To disable them, export `SYMAI_WARNINGS=0` in your environment variables.

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
