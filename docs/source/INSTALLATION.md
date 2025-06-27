# Installation

## Core Features

To get started with SymbolicAI, you can install it using pip:

```bash
pip install symbolicai
```

### Setting up a neurosymbolic API Key

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

### Optional Features

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

### Additional Requirements

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
