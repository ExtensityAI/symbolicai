# Plugins and External Packages

SymbolicAI is extended with **plugins**: ordinary Python packages that ship one or more
`Expression` classes and advertise them through the `symai.expressions`
[entry-point group](https://packaging.python.org/en/latest/specifications/entry-points/).
Once a plugin is installed in your environment, you load it by name with `Interface`.

> The previous bespoke loader — `Import("owner/repo")`, the `package.json` manifest, and
> the `sympkg`/`symdev` CLIs — has been removed. Plugins are now distributed and installed
> like any other Python package.

## Authoring a plugin

A plugin is a normal package. Declare each exported expression under the
`symai.expressions` entry-point group in its `pyproject.toml`, mapping a name to a
`module:Class` target:

```toml
[project.entry-points."symai.expressions"]
my_expression = "my_package.expressions:MyExpression"
```

`MyExpression` is a standard `symai.Expression` subclass. Declare runtime dependencies in
`[project.dependencies]` as usual — there is no separate manifest or `requirements.txt`.

## Installing a plugin

Install it like any package — from PyPI or directly from Git:

```bash
pip install my-symai-plugin
# or straight from a repository
pip install git+https://github.com/<owner>/<repo>
```

## Loading a plugin

Load it by its entry-point name:

```python
from symai import Interface

expr = Interface("my_expression")
result = expr(...)
```

`Interface(name)` resolves built-in interfaces first, then installed `symai.expressions`
plugins (a built-in name shadows a plugin of the same name). An unknown name raises a
clear `ValueError`.
