# Installation

## Core Features

To get started with SymbolicAI, you can install it using pip:

```bash
pip install symbolicai
```

Before using SymbolicAI, you need to set up API keys for the various engines. By default, SymbolicAI uses OpenAI's neural engines.

To set the OpenAI API Keys use the following command:

```bash
# Linux / MacOS
export OPENAI_API_KEY="<OPENAI_API_KEY>"

# Windows (PowerShell)
$Env:OPENAI_API_KEY="<OPENAI_API_KEY>"

# Jupyter Notebooks
%env OPENAI_API_KEY=<OPENAI_API_KEY>
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

And export the following API keys:

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

## Configuration File

You can specify engine properties in a `symai.config.json` file in your project path. This will replace the environment variables. For example:

```json
{
    "NEUROSYMBOLIC_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "NEUROSYMBOLIC_ENGINE_MODEL": "text-davinci-003",
    "SYMBOLIC_ENGINE_API_KEY": "<WOLFRAMALPHA_API_KEY>",
    "EMBEDDING_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "EMBEDDING_ENGINE_MODEL": "text-embedding-ada-002",
    "IMAGERENDERING_ENGINE_API_KEY": "<OPENAI_API_KEY>",
    "VISION_ENGINE_MODEL": "openai/clip-vit-base-patch32",
    "SEARCH_ENGINE_API_KEY": "<SERP_API_KEY>",
    "SEARCH_ENGINE_MODEL": "google",
    "OCR_ENGINE_API_KEY": "<APILAYER_API_KEY>",
    "SPEECH_TO_TEXT_ENGINE_MODEL": "base",
    "TEXT_TO_SPEECH_ENGINE_MODEL": "tts-1",
    "INDEXING_ENGINE_API_KEY": "<PINECONE_API_KEY>",
    "INDEXING_ENGINE_ENVIRONMENT": "us-west1-gcp",
    "COLLECTION_DB": "ExtensityAI",
    "COLLECTION_STORAGE": "SymbolicAI",
    "SUPPORT_COMMUNITY": false
}
```

With these steps completed, you should be ready to start using SymbolicAI in your projects.

> [NOTE]: Our framework allows you to support us train models for local usage by enabling the data collection feature. On application startup we show the terms of services and you can activate or disable this community feature. We do not share or sell your data to 3rd parties and only use the data for research purposes and to improve your user experience. To change this setting you will be prompted with in our setup wizard to enable or disable community support or you can go to the `symai.config.json` file located in your home directory of your `.symai` folder (i.e., `~/.symai/symai.config.json`), and turn it on/off by setting the `SUPPORT_COMMUNITY` property to `True/False` via the config file or the respective environment variable.
> [NOTE]: By default, the user warnings are enabled. To disable them, export `SYMAI_WARNINGS=0` in your environment variables.