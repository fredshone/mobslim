"""Sphinx configuration: ``uv run --group docs sphinx-build docs docs/_build``."""

project = "MobSlim"
author = "Fred Shone"
copyright = "2025-2026, Fred Shone"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "_generated"]  # _generated: tables included by pages
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]

autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_typehints = "description"
# optional extras: document the API even when they are not installed
autodoc_mock_imports = ["pyarrow", "mobslim_router", "mobslim_hermes", "mobslim_psim", "mobslim_dsim", "pymetis"]
default_role = "code"
suppress_warnings = ["myst.xref_missing", "ref.python"]  # ref.python: ambiguous `type`/re-exported names

html_theme = "furo"
html_title = "MobSlim"
html_static_path = ["_static"]
html_css_files = ["showcase/showcase.css"]
