"""Every import inside `symai/` must respect the layer boundary it crosses.

The layering was previously guarded by rules scattered across two tests, each keyed to a
*directory name* rather than to the tier it meant, with disagreeing forbidden-lists. The
gaps were not theoretical: adding `import symai.symbol` to `symai/providers/_client/` —
a direct violation of the constraint that a client never knows the framework — passed the
entire suite, because the one test watching that directory only grepped for the prefix
`symai.runtime`, and `symai/symbol.py` imports nothing, so no runtime module ever loaded.

This states the contract once, over layers instead of names. A module's layer comes from
its path, so a new module classifies itself and needs no edit here. `ALLOWED` enumerates
layers — a design decision that changes only when the architecture does — and a diff to it
is exactly the review signal that should be hard to miss.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "symai"

# Discovered, never listed. A hardcoded tuple of provider names is how the previous guards
# failed: a fourth provider was not in the list, so it classified as some other layer and
# silently opted out of every rule here — the violation the guard exists to catch passed
# green. A provider is a directory; asking the filesystem means a new one is covered on the
# day it is created rather than on the day someone remembers this file.
PROVIDERS = tuple(
    sorted(
        path.name
        for path in (PACKAGE / "providers").iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
)

# Which layers each layer may import. Reflexive entries are explicit rather than implied,
# so that a closed tier (one that may reach nothing but itself) is visible as such.
ALLOWED: dict[str, frozenset[str]] = {
    # Closed. The runtime defines the protocols providers implement and must never know
    # which providers exist, or configuration becomes a lie.
    "runtime": frozenset({"runtime"}),
    # Closed. A client is a faithful binding to one API and must stay usable without symai
    # — so it may not reach the runtime, the user layer, or another provider.
    "client:shared": frozenset({"client:shared"}),
    "client:provider": frozenset({"client:shared", "client:provider"}),
    # An engine abstracts over its client. It may not name the client tier at all; the
    # single exception is listed in CROSSING_EXCEPTIONS and has to earn it.
    "engine:shared": frozenset({"engine:shared", "runtime"}),
    "engine:provider": frozenset(
        {"engine:shared", "engine:provider", "client:shared", "client:provider", "runtime"}
    ),
    # The composition root: the one place allowed to know a client and an engine at once,
    # because wiring them together is its whole purpose.
    "loading": frozenset(
        {
            "loading",
            "engine:shared",
            "engine:provider",
            "client:shared",
            "client:provider",
            "runtime",
        }
    ),
    "contract": frozenset({"contract", "runtime", "user"}),
    "user": frozenset({"user", "runtime"}),
}

# `mapping.py` is the anti-corruption layer: translating the client's error hierarchy into
# the runtime's is its entire purpose, so it is the one engine module that must name both
# sides. Keeping the exception to a single named module means a second one has to be
# argued for in a diff rather than absorbed silently.
CROSSING_EXCEPTIONS: dict[str, frozenset[str]] = {
    "symai.providers._engine.mapping": frozenset({"client:shared"}),
}


def _classify(module: str) -> tuple[str, str | None]:
    """Map a dotted `symai.*` module to its layer and the provider that owns it, if any."""
    match module.split("."):
        case ["symai", "runtime", *_]:
            return ("runtime", None)
        case ["symai", "contract", *_]:
            return ("contract", None)
        case ["symai", "providers", "_client", *_]:
            return ("client:shared", None)
        case ["symai", "providers", "_engine", *_]:
            return ("engine:shared", None)
        case ["symai", "providers", provider, "client", *_] if provider in PROVIDERS:
            return ("client:provider", provider)
        case ["symai", "providers", provider, "engines", *_] if provider in PROVIDERS:
            return ("engine:provider", provider)
        case ["symai", "providers", provider, "loading"] if provider in PROVIDERS:
            return ("loading", provider)
        case ["symai", "loading"]:
            return ("loading", None)
        case _:
            return ("user", None)


def _imports(path: Path) -> list[str]:
    """Every `symai.*` module `path` imports, with relative imports resolved to absolute.

    Relative imports are resolved rather than skipped even though ruff's TID252 bans them:
    a guard whose coverage silently depends on a lint rule in a different config file is
    one relaxed rule away from passing vacuously.
    """
    package = ".".join(path.relative_to(ROOT).parent.parts)
    targets: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                targets.append(node.module or "")
                continue

            anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
            targets.append(".".join([*anchor, node.module] if node.module else anchor))

    return [target for target in targets if target == "symai" or target.startswith("symai.")]


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_no_import_crosses_a_layer_boundary_it_may_not() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        module = _module_name(path)
        source_layer, source_provider = _classify(module)
        allowed = ALLOWED[source_layer] | CROSSING_EXCEPTIONS.get(module, frozenset())

        for target in _imports(path):
            target_layer, target_provider = _classify(target)
            if target_layer not in allowed:
                violations.append(
                    f"{module} -> {target} ({source_layer} may not import {target_layer})"
                )
                continue

            # A provider's client and engines are private to that provider: sharing between
            # two providers has to go through the shared tier, or the rule-of-three decision
            # about when to extract a common base gets made by accident.
            if source_provider and target_provider and source_provider != target_provider:
                violations.append(
                    f"{module} -> {target} ({source_provider} may not import {target_provider})"
                )

    assert violations == []


def test_the_contract_classifies_every_layer_it_claims_to_govern() -> None:
    """A contract that classifies nothing passes for the wrong reason.

    Both halves matter. `_classify` falls through to "user", so a layer missing from
    `ALLOWED` would be treated as the user layer rather than flagged; and if provider
    discovery ever returned nothing, every provider module would fall through to "user"
    too — which permits importing the runtime, so the rule this file exists to enforce
    would pass vacuously against a tree that violates it everywhere.
    """
    layers = {_classify(_module_name(path))[0] for path in PACKAGE.rglob("*.py")}

    assert layers <= set(ALLOWED)
    assert layers >= {
        "client:provider",
        "client:shared",
        "engine:provider",
        "engine:shared",
        "loading",
        "runtime",
        "user",
    }
