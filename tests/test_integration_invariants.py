"""Anti-foot-gun invariants for the engine-redesign integration (Phase 0 · T0.3).

These tests turn the audit's *traps* — the properties a later simplification/dedup pass
could silently break — from a memo into failing CI. Rationale and sequencing live in
``docs/fullreport/09-INTEGRATION.md`` (Phase 0) and ``docs/fullreport/00-SUMMARY.md``.

Each invariant below is currently TRUE against the tree; if a future change violates one,
this file fails *before* the regression ships. Every field set is pinned deliberately —
changing a pin must be a conscious edit, never a side effect.

Guarded traps:
  * multimodal (image) request path is wired end-to-end through every language engine;
  * the provider *client* layer stays a faithful API binding that never imports ``symai.runtime``;
  * no ``TokenUsage`` / ``RateLimitMetadata`` field is dropped, and every field is producer-backed
    (including the single-provider fields that look dead from one provider's vantage point);
  * N-output / ``output_index`` is retained (spec-ratified; not collapsed to one output).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import symai.decoding as decoding
import symai.runtime.models as models

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "symai"
PROVIDERS = PACKAGE / "providers"

ENGINE_FILES = sorted(PROVIDERS.glob("*/engines/*.py"))
CLIENT_FILES = sorted(PROVIDERS.glob("*/client/**/*.py"))


def _call_kwarg_names(paths: list[Path]) -> set[str]:
    """Every keyword-argument name used in a call across ``paths`` (a producer signal)."""
    names: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg is not None:
                names.add(node.arg)
    return names


def _absolute_imports(path: Path) -> set[str]:
    """Absolute dotted module names imported by ``path``, resolving relative imports."""
    module_parts = list(path.relative_to(ROOT).with_suffix("").parts)
    package_parts = module_parts[:-1]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    imports.add(node.module)
            else:
                base = package_parts[: len(package_parts) - (node.level - 1)]
                tail = [node.module] if node.module else []
                resolved = ".".join(base + tail)
                if resolved:
                    imports.add(resolved)
    return imports


# --- Invariant 1: the multimodal (image) path is wired end-to-end --------------------


def test_image_content_type_exists() -> None:
    """The normalized contract must keep an image content type (do not delete as 'unused')."""
    assert models.ContentType.IMAGE == "image"
    assert models.ImageContent.__name__ == "ImageContent"


def test_every_language_engine_wires_image_content() -> None:
    """Each language engine must still reference ``ImageContent`` — the multimodal request path."""
    language_engines = [p for p in ENGINE_FILES if p.name not in {"__init__.py", "embedding.py"}]
    assert language_engines, "no language engine modules discovered"
    missing = [
        str(p.relative_to(PACKAGE))
        for p in language_engines
        if "ImageContent" not in p.read_text(encoding="utf-8")
    ]
    assert not missing, f"language engines dropped the image/multimodal path: {missing}"


# --- Invariant 2: the client layer is a provider-pure API binding ---------------------


def test_provider_clients_never_import_runtime() -> None:
    """A client is a faithful HTTP/API binding; it must never know about ``symai.runtime``.

    The engine (adapter) is the only crossing point between a provider client and the runtime.
    """
    assert CLIENT_FILES, "no provider client modules discovered"
    offenders = {
        str(path.relative_to(PACKAGE)): sorted(
            m for m in _absolute_imports(path) if m == "symai.runtime" or m.startswith("symai.runtime.")
        )
        for path in CLIENT_FILES
    }
    leaking = {path: mods for path, mods in offenders.items() if mods}
    assert not leaking, f"provider client modules import runtime: {leaking}"


# --- Invariant 3: usage / rate-limit fields are pinned and producer-backed -------------

# Pinned deliberately. Adding or removing a field is a conscious edit, not a side effect.
EXPECTED_TOKEN_USAGE_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
        "cache_miss_prompt_tokens",  # DeepSeek-only producer
        "reasoning_tokens",
        "image_tokens",  # Cerebras-only producer
        "accepted_prediction_tokens",  # Cerebras-only producer
        "rejected_prediction_tokens",  # Cerebras-only producer
    }
)
EXPECTED_RATE_LIMIT_FIELDS = frozenset(
    {
        "limit_requests_day",
        "limit_tokens_minute",
        "remaining_requests_day",
        "remaining_tokens_minute",
        "reset_requests_day",
        "reset_tokens_minute",
    }
)  # all Cerebras-only producers today


def test_token_usage_field_set_is_pinned() -> None:
    assert set(models.TokenUsage.model_fields) == EXPECTED_TOKEN_USAGE_FIELDS


def test_rate_limit_field_set_is_pinned() -> None:
    assert set(models.RateLimitMetadata.model_fields) == EXPECTED_RATE_LIMIT_FIELDS


def test_every_usage_field_has_a_producing_engine() -> None:
    """No usage field is dead: each is assigned by at least one engine (as a call kwarg).

    Single-provider fields (``image_tokens``, ``cache_miss_prompt_tokens``, ...) look unused
    from any one provider — this guard prevents a dedup pass from deleting them.
    """
    produced = _call_kwarg_names(ENGINE_FILES)
    unproduced = sorted(field for field in models.TokenUsage.model_fields if field not in produced)
    assert not unproduced, f"TokenUsage fields with no engine producer: {unproduced}"


def test_every_rate_limit_field_has_a_producing_engine() -> None:
    produced = _call_kwarg_names(ENGINE_FILES)
    unproduced = sorted(field for field in models.RateLimitMetadata.model_fields if field not in produced)
    assert not unproduced, f"RateLimitMetadata fields with no engine producer: {unproduced}"


# --- Invariant 4: N-output / output_index is retained (spec-ratified) ------------------


def test_decode_output_keeps_output_index() -> None:
    assert "output_index" in inspect.signature(decoding.decode_output).parameters


def test_response_models_keep_indexed_multi_output() -> None:
    """``LanguageModelResponse.outputs`` is a tuple of indexed outputs — do not collapse to one."""
    assert "outputs" in models.LanguageModelResponse.model_fields
    assert "index" in models.LanguageModelOutput.model_fields
    outputs_annotation = models.LanguageModelResponse.model_fields["outputs"].annotation
    assert "tuple" in repr(outputs_annotation).lower()
