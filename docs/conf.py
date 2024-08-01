# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Evariste"
copyright = "2022, Evariste"
author = "Evariste"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    # third party
    "enum_tools.autoenum",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_mock_imports = [
    "zmq",
    "pandas",
    "typeguard",
    "numexpr",
    "youtokentome",
    "matplotlib",
    "scipy",
    "pynvml",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

add_module_names = False
todo_include_todos = True


import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath("./../formal"))

# # Ensure that the __init__ method gets documented.
# def skip(app, what, name, obj, skip, options):
#     if name == "__init__":
#         return False
#     return skip


# def setup(app):
#     app.connect("autodoc-skip-member", skip)
