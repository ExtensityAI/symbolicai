# Workflows

## Engine testing

When asked to test an engine, run both request-shape tests and real provider checks.

1. Identify the engine family and provider from the changed files or the user request.
2. Pick the matching API key from `api_keys.log`. Do not print the key, paste it into chat, or commit it.
3. Update the active local config, usually `.venv/.symai/symai.config.json`, with the matching engine model and API key pair.
4. Run the mock pytest path first:

```bash
uv run pytest tests/engines --engine-api=mock
```

5. Run the live pytest path for the same engine or narrow test file:

```bash
uv run pytest tests/engines --engine-api=live
```

6. Run live terminal Python checks for provider features that pytest may not cover. Pick checks that match the engine:
   - plain text generation
   - function calling or tool calling
   - vision or multimodal input
   - embeddings
   - search citations
   - scraping
   - formal or symbolic calls

Use `uv run python` for these checks so the current project environment is used. Use files from `tests/data/` when the engine needs input data, such as `sample.jpg`, `sample.png`, `sample.pdf`, `sample.txt`, `sample.html`, `sample.mp3`, `sample.docx`, `sample.xlsx`, or `sample.zip`.

7. Return evidence, not a claim. Include:
   - exact pytest commands and pass, skip, or fail counts
   - exact terminal Python commands or a short description of the script
   - provider response facts, such as status code, non-empty text, tool-call name, image answer, embedding dimension, citation count, or parsed output type
   - the configured model name, but never the API key

# Testing

Engine API tests use the same test files for mock and live runs.

Mock mode validates typed request shapes and fake responses:

```bash
uv run pytest tests/engines --engine-api=mock
```

Live mode uses `symai.config.json` for the configured model and API key:

```bash
uv run pytest tests/engines --engine-api=live
```

Run the DeepSeek request-interface tests:

```bash
uv run pytest tests/engines/neurosymbolic/test_deepseek_request_interface.py
```

Run only the live DeepSeek check:

```bash
uv run pytest tests/engines/neurosymbolic/test_deepseek_request_interface.py::test_deepseek_live_request_interface --engine-api=live
```

Collect without running:

```bash
uv run pytest tests --engine-api=mock --collect-only
```

# Engine request API

Expected engine flow:

```text
prepare(argument)
build_request(argument)
call_request(request)
parse_response(response)
```

Useful checks while editing an engine:

```bash
uv run ruff format symai/backend/mixin/deepseek.py symai/backend/engines/neurosymbolic/engine_deepseekX_reasoning.py
uv run ruff check symai/backend/mixin/deepseek.py symai/backend/engines/neurosymbolic/engine_deepseekX_reasoning.py --output-format concise
uv run python -m ast symai/backend/mixin/deepseek.py >/dev/null
```
