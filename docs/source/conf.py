# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Megabouts'
copyright = '2024, Adrien Jouary'
author = 'Adrien Jouary'
release = '0.1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autosectionlabel",
              "nbsphinx",
              "sphinx.ext.viewcode",
              "sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinxcontrib.video"]

templates_path = ['_templates']
exclude_patterns = []

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = ["css/custom.css"]
html_favicon = "_static/favicon.ico"
html_context = {"default_mode": "dark"}
html_theme_options = {
    "navbar_end": []#["navbar-icon-links.html", "search-field.html"]
}
