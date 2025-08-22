# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import re
sys.path.insert(0, os.path.abspath('..'))
from specula.__version__ import __version__

project = 'SPECULA'
copyright = '2025, Fabio Rossi, Alfio Puglisi, Guido Agapito'
author = 'Fabio Rossi, Alfio Puglisi, Guido Agapito'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx_rtd_theme',        # ReadTheDocs theme
              'myst_parser',
              'nbsphinx',
              'numpydoc',
              ]

source_suffix = {
    '.rst': None,
    '.md': None,
}

intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    # 'matplotlib': ('http://matplotlib.org/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    }


templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'index'

# General substitutions.

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
version = re.sub(r'\.dev-.*$', r'.dev', __version__)
release = __version__

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


print("%s (VERSION %s)" % (project, version))

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme = 'pydata_sphinx_theme'

#html_theme_options = {
#  "show_nav_level": 2
#}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'py:obj'
