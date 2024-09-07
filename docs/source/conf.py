import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

#  Configuration file for the Sphinx documentation builder.
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
#"nbsphinx",
extensions = ["sphinx.ext.autosectionlabel",
              "myst_nb",
              "sphinx.ext.viewcode",
              "sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinxcontrib.video",
              "sphinx_togglebutton"]


templates_path = ['_templates']
exclude_patterns = []

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_attr_annotations = True

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = ["css/custom.css"]
html_favicon = "_static/favicon.ico"
html_context = {"default_mode": "dark"}

# For plotly
html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"]

'''
# Removes sidebars for all pages
html_sidebars = {"**":[]} 
html_theme_options = {
    "navbar_start": [],  # Empty the navbar start section if needed
    "navbar_end": [],    # Empty the navbar end section if needed
    "footer_items": [],  # Empty the footer section if needed
    "secondary_sidebar_items": [],  # Remove secondary sidebar items
    "left_sidebar": False,  # Explicitly disable the left sidebar
    "right_sidebar": False, # Explicitly disable the right sidebar
    "navigation_depth": 1,  # Set navigation depth to minimal if needed
}'''
