# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys
from datetime import datetime


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.abspath("../.."))


def _resolve_version() -> str:
    try:
        import zenreg

        return getattr(zenreg, "__version__", "0.0.0+unknown")
    except Exception:
        return "0.0.0+unknown"


project = "ZenReg"
author = "Fabrizio Musacchio"
copyright = f"{datetime.now().year}, {author}"
release = _resolve_version()

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 5,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
html_static_path = ["_static"]
html_logo = "_static/ZenReg_logo_square2_transparent.png"
html_css_files = ["custom.css"]

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

copybutton_selector = "div.highlight pre"
