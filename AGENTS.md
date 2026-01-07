# Repository Guidelines

Guidance for coding agents and contributors.

## Architecture Overview
SymbolicAI is a neuro-symbolic Python framework. `Symbol`/`Expression` route work to pluggable engines (LLMs, search,
scrape, vector DBs) configured via `symai.config.json`.

### Key Components
- `symai/symbol.py` — `Symbol` primitive (syntactic vs semantic view).
- `symai/strategy.py` — contract system (`@contract`) and validation/retry flow.
- `symai/backend/engines/` — provider implementations (`neurosymbolic/`, `search/`, `scrape/`, `index/`, …).
- `symai/extended/interfaces/` — higher-level interfaces that select/compose engines.
- `symai/endpoints/`, `symai/server/` — FastAPI endpoints and optional local servers.

## Project Structure
- `symai/` — library source (import path `symai`).
- `tests/` — pytest suite.
- `docs/` — documentation sources; `examples/` — runnable examples.
- `assets/`, `public/` — static files; `scripts/` — dev utilities (not public API).

## Setup, Lint, and Tests
Environment (Python 3.10+), using `uv`:

```bash
uv sync --python 3.10
source .venv/bin/activate
```

Extras / locked installs:

```bash
uv sync --frozen
uv sync --extra scrape   # or: --extra all
```

Lint/format (Ruff; `ruff.toml`):

```bash
ruff check symai --config ruff.toml
ruff format symai
```

Tests:

```bash
pytest
pytest -m mandatory
pytest -q --tb=no tests/test_imports.py
```

CLI entrypoints (after install): `symchat`, `symsh`, `symconfig`, `symserver`.

## Configuration & Secrets
- Config precedence: `./symai.config.json`, `{venv}/.symai/symai.config.json`, then `~/.symai/symai.config.json`.
- Common keys: `NEUROSYMBOLIC_ENGINE_MODEL`, `NEUROSYMBOLIC_ENGINE_API_KEY`, `SYMAI_WARNINGS=0`, `SUPPORT_COMMUNITY`.
- Inspect active config with `symconfig`.
- Never commit API keys, tokens, or generated artifacts (`dist/`, caches, logs).

## Style Notes
- 4-space indentation, 100-char lines, double quotes (`ruff.toml`).
- Prefer clear names (`snake_case` functions/modules, `CapWords` classes, `UPPER_SNAKE_CASE` constants).

## Agent Workflow Rules (Project-Specific)
- Ask for clarification instead of guessing on ambiguous behavior or invariants.
- Treat type hints as contracts; do not add runtime type checks except at trust boundaries (CLI/env, JSON/network, disk).
- Prefer minimal diffs; edit existing code over adding new files unless necessary.
- Do not add/modify `tests/` or run tests unless explicitly requested; if requested, run the narrowest relevant `pytest` command.
- When you change Python files outside `tests/`: run `ruff check <changed_files> --output-format concise --config ruff.toml` and fix issues.
- Keep search local-first (`rg`); follow imports instead of repo-wide “random scanning”.
- If adding a regex, include a short comment explaining what it matches.
- Update `TODO.md` when tasks are completed, added, or re-scoped.

## Commit & PR Notes
- Commit messages commonly use `(<type>) <summary>`: `(feat)`, `(fix)`, `(chore)`, `(feature)`.
- PRs: describe intent + user impact, link issues, note config/env var changes, and include validation commands + results.
