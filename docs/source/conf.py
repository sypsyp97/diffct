# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../..'))

# The docs build has no GPU. Mock the CUDA stack so that autodoc can import diffct.
autodoc_mock_imports = ['numba', 'torch']

project = 'diffct'
copyright = '2025-2026, Yipeng Sun'
author = 'Yipeng Sun'
_init = (Path(__file__).parents[2] / 'diffct' / '__init__.py').read_text()
release = re.search(r"__version__ = '([^']+)'", _init).group(1)
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []
autodoc_typehints = 'description'
copybutton_prompt_text = r'\$ |>>> '
copybutton_prompt_is_regexp = True

html_theme = 'furo'
html_title = f'diffct {release}'
html_static_path = []
html_theme_options = {
    'source_repository': 'https://github.com/sypsyp97/diffct/',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
}
