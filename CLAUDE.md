# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SymbolicAI is a neuro-symbolic framework (package name: `symbolicai`, import as `symai`) that bridges classical Python programming with LLMs. The core abstraction is the `Symbol` — a value container that operates in two modes: **syntactic** (normal Python) and **semantic** (wired to an LLM engine). Version is defined in `symai/__init__.py` as `SYMAI_VERSION`.

## Common Commands

```bash
# Install (editable, with dev tools)
uv sync --all-extras --dev

# Lint (ruff config in ruff.toml — 100 char line limit, Python 3.10 target)
ruff check symai/ --output-format concise --config ruff.toml
ruff check symai/some_file.py --output-format concise --config ruff.toml  # single file

# Autofix lint
ruff check symai/ --fix --config ruff.toml

# Format
ruff format symai/ --config ruff.toml

# Run all tests (verbose, ignores test_imports.py by default — see pytest.ini)
pytest tests

# Run mandatory tests only
pytest -m mandatory

# Run a single test file or specific test
pytest tests/test_symbol.py
pytest tests/test_primitives.py::test_name
```

## Architecture

### Class Hierarchy

```
Symbol (symai/symbol.py)
  └── Expression (symai/symbol.py)        — lazy-evaluated; subclass and implement forward()
        └── TrackerTraceable (symai/components.py)
              └── Function (symai/components.py)  — prompt-driven LLM call with pre/post processors
```

- **Symbol**: The fundamental value wrapper. Supports dual-mode operations (syntactic via Python operators, semantic via LLM). All operator overloads (`__add__`, `__eq__`, comparisons, etc.) dispatch through `core.py` primitives.
- **Expression**: Adds lazy evaluation via `forward()`. Subclass this for custom neuro-symbolic operations. Calling an Expression executes `forward()` and stores the result.
- **Function**: An Expression specialized for prompt-based LLM calls. Configurable with `pre_processors`, `post_processors`, `constraints`, and `return_type`.

### Decorators (`symai/core.py`)

`@zero_shot(prompt=...)` and `@few_shot(prompt=..., examples=...)` turn methods into LLM-backed operations. They wrap the decorated method with a `Function` under the hood, attaching pre/post processors and constraints.

### Engine System (`symai/backend/`)

Engines are the LLM/service backends. All inherit from `Engine` (abstract base in `backend/base.py`). The `forward(argument)` method is the contract — it receives a structured argument and returns `(result, metadata)`.

**Provider mixins** (`backend/mixin/`): `OpenAIMixin`, `AnthropicMixin`, `GoogleMixin`, `DeepSeekMixin` — shared API client logic mixed into concrete engine classes.

**Neurosymbolic engines** (`backend/engines/neurosymbolic/`): The primary LLM engines. Naming pattern: `engine_{provider}_{model_family}_{mode}.py`. Examples:
- `engine_openai_gptX_chat.py` → `GPTXChatEngine(Engine, OpenAIMixin)`
- `engine_anthropic_claudeX_reasoning.py` → `ClaudeXReasoningEngine(Engine, AnthropicMixin)`
- `engine_google_geminiX_reasoning.py` → `GeminiXReasoningEngine(Engine, GoogleMixin)`

**Specialized engines** (`backend/engines/`): drawing, embedding, search, scrape, index, OCR, speech_to_text, text_to_speech, text_vision, symbolic (WolframAlpha), execute (Python), files, userinput, lean.

**EngineRepository** (`symai/functional.py`): Singleton registry. Engines register via `EngineRepository.register(id, engine_instance)` or dynamically from plugins/packages. The active engine is selected based on `symai.config.json` settings.

### Contracts (`symai/strategy.py`)

The `@contract` decorator implements Design by Contract for Expression subclasses:
- Define `pre()`, `forward()`, `post()` methods on an Expression
- The decorator validates pre/post conditions and auto-retries with remedy logic on failure
- Uses `LLMDataModel` (Pydantic-based, `symai/models/base.py`) for structured input/output schemas
- Default retry: 8 tries with exponential backoff

### Configuration System (`symai/backend/settings.py`)

Three config files, resolved by priority: **Debug** (CWD) > **Environment** (`{sys.prefix}/.symai/`) > **Home** (`~/.symai/`).

| File | Purpose |
|---|---|
| `symai.config.json` | API keys, engine models, all engine settings |
| `symsh.config.json` | Shell UI colors, plugin settings |
| `symserver.config.json` | Local server state (llama.cpp, Qdrant, HuggingFace) |

`SymAIConfig` (singleton at `symai.config_manager`) handles loading, migration, and path resolution.

### Processing Pipeline

Requests flow through: **PreProcessors** → **Engine.forward()** → **PostProcessors**

- `symai/pre_processors.py`: Transform input before engine call (e.g., `JsonPreProcessor`)
- `symai/post_processors.py`: Transform output after engine call (e.g., `StripPostProcessor`, `CodeExtractPostProcessor`, `JsonTruncatePostProcessor`)
- `symai/processor.py`: `ProcessorPipeline` chains processors

### Key Modules

| Module | Role |
|---|---|
| `symai/prompts.py` | Prompt templates, `PromptRegistry`, `PromptLanguage` |
| `symai/strategy.py` | `@contract` decorator, retry/remedy strategies |
| `symai/components.py` | `Function`, `GraphViz`, and built-in Expression subclasses |
| `symai/imports.py` | Dynamic module loading via `Import` |
| `symai/extended/` | Higher-level tools: `Conversation`, document handling, API builder, solver, vectordb |
| `symai/models/base.py` | `LLMDataModel` — Pydantic BaseModel subclass formatted for LLM prompts |
| `symai/ops/` | Operator/primitive dispatch tables (`SYMBOL_PRIMITIVES`) |

### CLI Entry Points (defined in `pyproject.toml [project.scripts]`)

| Command | Module | Purpose |
|---|---|---|
| `symchat` | `symai.chat:run` | Interactive chat |
| `symsh` | `symai.shell:run` | Neuro-symbolic shell |
| `sympkg` | `symai.extended.packages.sympkg:run` | Package manager |
| `symdev` | `symai.extended.packages.symdev:run` | Dev tools |
| `symrun` | `symai.extended.packages.symrun:run` | Run expressions |
| `symconfig` | `symai:display_config` | Inspect active config |
| `symserver` | `symai:run_server` | Start local server |

## Conventions

- **Python >= 3.10** required; ruff targets `py310`
- **Line length**: 100 characters
- **Ruff rules**: see `ruff.toml` for the full selected set; notably `tests/` and `scripts/` are excluded from linting
- **Imports**: `symai/__init__.py` runs `_start_symai()` at import time (reads config, validates API keys) — this means `import symai` has side effects and requires a valid config
- **Engine pattern**: new engines subclass `Engine`, mix in a provider mixin if applicable, implement `forward()`, and register via `EngineRepository`
- **Version bumps**: update `SYMAI_VERSION` in `symai/__init__.py`; `setuptools-scm` reads it via `tool.setuptools.dynamic.version`

## Memories

- When adding new dependencies to `pyproject.toml`, check if the PyPI package name differs from the Python import name. If it does, add an entry to the `_IMPORT_NAME` mapping in `tests/test_imports.py`. The test reads `pyproject.toml` dynamically, but non-obvious mappings (e.g., `scikit-learn` → `sklearn`) must be registered manually in that dict.
