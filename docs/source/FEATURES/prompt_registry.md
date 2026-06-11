# Prompt Registry

The `PromptRegistry` is a singleton that manages Jinja2 templates and an optional TOML manifest. It replaces the previous enum-based language/model system with a file-driven approach: you organize prompts as `.jinja` files in a folder, optionally alongside a `manifest.toml`, and load them in one call.

## Folder Layout

```
prompts/
├── manifest.toml
├── system.jinja
├── task.jinja
└── stages/
    ├── plan.jinja
    └── review.jinja
```

Every `.jinja` file is registered under a **dot-separated key** derived from its path relative to the root folder. Subfolders become dotted segments:

| File | Key |
|------|-----|
| `system.jinja` | `system` |
| `task.jinja` | `task` |
| `stages/plan.jinja` | `stages.plan` |
| `stages/review.jinja` | `stages.review` |

Load the folder with:

```python
from symai import PromptRegistry

registry = PromptRegistry()
registry.load_from_folder("prompts")
```

## The manifest.toml

The manifest serves two purposes:

1. **Template data** — every value defined in the manifest is accessible inside any `.jinja` template through the `manifest` object.
2. **Structured metadata** — because it is TOML, you can use sections, typed values, arrays, and inline tables to organize versioning, authorship, feature flags, or any other configuration your prompts depend on.

### Example manifest

```toml
version = "1.0.0"
author = "Foo Bar"

[features]
max_iterations = 5
min_confidence = 0.8
style = "concise"

[features.output]
max_sections = 3
min_facts_per_section = 2
```

### Accessing manifest values in templates

Inside any `.jinja` file the manifest is available as the `manifest` variable:

```jinja
# Task

You are an agent operating in {{ mode }} mode.

## Constraints

- Maximum iterations: {{ manifest.features.max_iterations }}
- Minimum confidence: {{ manifest.features.min_confidence }}
- Output style: {{ manifest.features.style }}

{% if manifest.features.output %}
## Output Format

- Sections: up to {{ manifest.features.output.max_sections }}
- Facts per section: at least {{ manifest.features.output.min_facts_per_section }}
{% endif %}

## Input

{{ input }}
```

The `tojson` filter serializes any manifest value (including nested sections and Pydantic models) to a JSON string.

## Rendering

Call `render()` with the template key and any runtime variables:

```python
output = registry.render("task", mode="research", input=user_query)
```

Runtime variables are passed alongside `manifest` into the template context. The same applies to templates in subfolders — e.g. `registry.render("stages.plan", input=plan_context)` will resolve `stages/plan.jinja` with the manifest and `input` available.

## Programmatic Registration

For cases where templates are generated at runtime rather than loaded from disk:

```python
registry.register_template("inline.greeting", "Hello, {{ name }}!")
registry.set_manifest({"version": "0.1.0", "features": {"style": "formal"}})

result = registry.render("inline.greeting", name="world")
```

Use `has_template(key)` to check whether a key exists before rendering.
