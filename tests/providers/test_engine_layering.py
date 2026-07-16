"""An engine abstracts over its client, so it must never learn what the transport is.

This is checked structurally rather than by grepping for `httpx`, because a name search
cannot see the shape. The engine base previously imported the client's `ResponseMetadata`
— a type carrying `status_code`, which exists only because the provider is reached over
HTTP — and a grep for `httpx` reported the layer clean while that seam was wide open.
An import is the thing that actually couples two modules, so an import is what is asserted
here.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED_ENGINE = ROOT / "symai" / "providers" / "_engine"
CLIENT_TIER = "symai.providers._client"

# `mapping.py` is the anti-corruption layer: translating the client's error hierarchy into
# the runtime's is its entire purpose, so it is the one engine module that must name both
# sides. Naming it here rather than exempting the whole package keeps the exception to a
# single reviewable module — adding a second one should require arguing for it in a diff.
CROSSING_MODULES = frozenset({"mapping.py"})


def _absolute_import_targets(path: Path) -> list[str]:
    """Every module `path` imports, with relative imports resolved to absolute.

    Relative imports are resolved rather than skipped even though ruff bans them: a guard
    whose coverage silently depends on a lint rule in a different config file is one
    relaxed rule away from passing vacuously.
    """
    package = ".".join(path.relative_to(ROOT).parent.parts)
    targets: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, node.module] if node.module else anchor)
            targets.append(base)

    return targets


def test_the_shared_engine_layer_names_no_transport() -> None:
    offenders = [
        f"{path.relative_to(ROOT)} -> {target}"
        for path in sorted(SHARED_ENGINE.rglob("*.py"))
        if path.name not in CROSSING_MODULES
        for target in _absolute_import_targets(path)
        if target == CLIENT_TIER or target.startswith(f"{CLIENT_TIER}.")
    ]

    assert offenders == []


def test_the_transport_agnostic_rule_is_checked_by_shape_not_by_spelling() -> None:
    """The client tier must stay HTTP-shaped, or the rule above proves nothing.

    If `ResponseMetadata` ever loses `status_code`, importing it into an engine would stop
    being a layering violation and the assertion above would be guarding a rule that no
    longer bites. That would make this suite quietly weaker, so it is pinned: the point is
    that the client tier speaks HTTP and the engine tier must not.
    """
    from symai.providers._client.transport import ResponseMetadata

    assert "status_code" in ResponseMetadata.model_fields
