"""SymbolicAI's built-in GitHub package loader has been removed.

Plugins are now standard Python packages discovered through the ``symai.expressions``
entry-point group and loaded by name with :class:`~symai.interfaces.Interface`. Install
one like any other package, e.g.::

    pip install git+https://github.com/<owner>/<repo>

then load it with ``Interface("<entry-point-name>")``.
"""


class Import:
    """Removed. SymbolicAI plugins are standard Python packages discovered via the
    ``symai.expressions`` entry-point group; load them with ``Interface("<name>")``.

    Kept importable for backwards compatibility, but construction always raises with
    migration guidance."""

    def __init__(self, module: str = "<owner>/<repo>", *_args, **_kwargs):
        msg = (
            "symai's built-in GitHub package loader was removed. Install the plugin as a "
            f"normal Python package, e.g. `pip install git+https://github.com/{module}`, then "
            'load it by its entry-point name with `Interface("<name>")` '
            "(the `symai.expressions` entry-point group)."
        )
        raise RuntimeError(msg)
