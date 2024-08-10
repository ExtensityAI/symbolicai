# Installing SymbolicAI

## Quick Install

To get started with SymbolicAI, you can install it using pip:

```bash
pip install symbolicai
```

## API Keys

Before using SymbolicAI, you need to set up API keys for the various engines. By default, SymbolicAI uses OpenAI's neural engines.

### Setting up OpenAI API Key

```bash
# Linux / MacOS
export OPENAI_API_KEY="<OPENAI_API_KEY>"

# Windows (PowerShell)
$Env:OPENAI_API_KEY="<OPENAI_API_KEY>"

# Jupyter Notebooks
%env OPENAI_API_KEY=<OPENAI_API_KEY>
```

### Optional API Keys

For additional functionality, you can set up the following API keys:

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

## Optional Dependencies

To use additional features, install the following extras:

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

## Additional Requirements

- **SpeechToText Engine**: Install `ffmpeg` for audio processing
- **WebCrawler Engine**: Chrome browser is required for Selenium

## Configuration File

You can specify engine properties in a `symai.config.json` file in your project path. This will replace the environment variables. Example:

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