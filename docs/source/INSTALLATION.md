## Installation

### Core Features

To get started with SymbolicAI, you can install it using pip:

```bash
pip install symbolicai
```

Alternatively, clone the repository and set up a Python virtual environment using uv:
```bash
git clone git@github.com:ExtensityAI/symbolicai.git
cd symbolicai
uv sync --python x.xx
source ./.venv/bin/activate
```
Running `symconfig` will now use this Python environment.

#### Optional Features

SymbolicAI uses multiple engines to process text, speech and images. We also include search engine access to retrieve information from the web. To use all of them, you will need to also install the following dependencies and assign the API keys to the respective engines. E.g.:

```bash
pip install "symbolicai[hf]",
pip install "symbolicai[llamacpp]",
pip install "symbolicai[bitsandbytes]",
pip install "symbolicai[wolframalpha]",
pip install "symbolicai[whisper]",
pip install "symbolicai[webscraping]",
pip install "symbolicai[serpapi]",
pip install "symbolicai[services]",
pip install "symbolicai[solver]"
```

Or, install all optional dependencies at once:

```bash
pip install "symbolicai[all]"
```

To install dependencies exactly as locked in the provided lock file:
```bash
uv sync --frozen
```

To install optional extras via uv:
```bash
uv sync --extra all # all optional extras
uv sync --extra webscraping # only webscraping
```

> ❗️**NOTE**❗️Please note that some of these optional dependencies may require additional installation steps. Additionally, some are only experimentally supported now and may not work as expected. If a feature is extremely important to you, please consider contributing to the project or reaching out to us.

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

Before using the `symai`, we recommend inspecting your current configuration setup using the command below. It will start the initial packages caching and initializing the `symbolicai` configuration files.

```bash
symconfig

# UserWarning: No configuration file found for the environment. A new configuration file has been created at <full-path>/.symai/symai.config.json. Please configure your environment.
```

You then must edit the `symai.config.json` file. A neurosymbolic engine is **required** to use the `symai` package. Read more about how to use a neuro-symbolic engine [here](https://extensityai.gitbook.io/symbolicai/engines/neurosymbolic_engine).

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
    "SPEECH_TO_TEXT_API_KEY": "",
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
