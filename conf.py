"""Configuration file for the Sphinx documentation builder."""
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import toml

with open("pyproject.toml") as f:
    data = toml.load(f)

# -- Project information -----------------------------------------------------

project = "SymbolicAI"
copyright = "2023, AlphaCore AI e.U."
author = "Marius-Constantin Dinu"

version = data["project"]["dynamic"][0]
release = version

html_title = project


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_panels",
    "IPython.sphinxext.ipython_console_highlighting",
]
source_suffix = [".py", ".md", ".rst"]

autodoc_pydantic_model_show_json = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_config_members = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_members = False
autodoc_pydantic_model_undoc_members = False
# autodoc_typehints = "signature"
# autodoc_typehints = "description"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "build", "dist", "docs", "output", "symbolicai.egg-info", "tests", ".vscode", "notebooks", "examples", "assets"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/ExtensityAI/symbolicai",
    "use_repository_button": True,
    "show_toc_level": 2,
    "header_links_before_dropdown": 5,
    "icon_links_label": "Quick Links",
    "primary_sidebar_end": ["indices.html"],
}

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "Extensity",  # Username
    "github_repo": "symbolicai",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

master_doc = 'index'
