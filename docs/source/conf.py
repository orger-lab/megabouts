import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath("../.."))
from megabouts import __version__

# -- Project information -----------------------------------------------------
project = "Megabouts"
copyright = "2024, Adrien Jouary"
author = "Adrien Jouary"
release = __version__
version = __version__

# -- Extensions configuration -----------------------------------------------
extensions = [
    "sphinx.ext.autosectionlabel",  # Allows referencing sections using labels
    "myst_nb",  # Jupyter notebook support
    "sphinx_design",  # Enhanced web components
    "sphinx.ext.viewcode",  # Links to source code
    "sphinx.ext.autodoc",  # API documentation from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinxcontrib.video",  # Video embedding support
    "sphinx_togglebutton",  # Collapsible content
    "sphinx_tabs.tabs",
]

# -- Path setup -----------------------------------------------------------
templates_path = ["_static"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Napoleon Settings ----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_attr_annotations = True

# -- AutoDoc Settings ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- HTML Theme Settings ------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/favicon.ico"
html_context = {"default_mode": "light"}

# Theme customization
html_theme_options = {
    # GitHub link in header
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/orger-lab/megabouts",
            "icon": "fa-brands fa-github",
        }
    ],
    # Navigation bar links
    "navbar_links": [
        {"name": "Installation", "url": "usage"},
        {"name": "Tutorials", "url": "tutorials"},
        {"name": "API", "url": "api/index"},
    ],
}

# -- Notebook Settings -------------------------------------------------
# Disable notebook execution during build
nb_execution_mode = "off"
jupyter_execute_notebooks = "off"

# Configure notebook output priority
nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    )
}
