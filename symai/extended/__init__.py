from importlib import import_module as _import_module

from .api_builder import APIBuilder, APIExecutor
from .bibtex_parser import BibTexParser
from .os_command import OSCommand
from .taypan_interpreter import TaypanInterpreter
from .vectordb import VectorDB

__all__ = [
    "APIBuilder",
    "APIExecutor",
    "BibTexParser",
    "OSCommand",
    "TaypanInterpreter",
    "VectorDB",
]

_seen_names = set(__all__)


def _export_module(module_name: str, seen_names: set[str] = _seen_names) -> None:
    module = _import_module(f"{__name__}.{module_name}")
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = [name for name in dir(module) if not name.startswith("_")]
    for name in public_names:
        globals()[name] = getattr(module, name)
        if name not in seen_names:
            __all__.append(name)
            seen_names.add(name)


for _module_name in [
    "arxiv_pdf_parser",
    "conversation",
    "document",
    "file_merger",
    "graph",
    "html_style_template",
    "packages",
    "repo_cloner",
    "solver",
    "summarizer",
]:
    _export_module(_module_name)


del _export_module
del _module_name
del _seen_names
