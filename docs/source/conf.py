# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import atexit
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
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []
autodoc_typehints = 'description'
autosummary_generate = True
copybutton_prompt_text = r'\$ |>>> '
copybutton_prompt_is_regexp = True

html_theme = 'pydata_sphinx_theme'
html_title = f'diffct {release}'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_favicon = '_static/favicon.svg'
html_logo = '_static/favicon.svg'
html_context = {
    'github_user': 'sypsyp97',
    'github_repo': 'diffct',
    'github_version': 'main',
    'doc_path': 'docs/source',
    'default_mode': 'auto',
}
html_theme_options = {
    'logo': {'text': 'diffct'},
    'github_url': 'https://github.com/sypsyp97/diffct',
    'icon_links': [
        {'name': 'PyPI', 'url': 'https://pypi.org/project/diffct/', 'icon': 'fa-brands fa-python'},
    ],
    'header_links_before_dropdown': 6,
    'navbar_align': 'left',
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'secondary_sidebar_items': ['page-toc', 'edit-this-page'],
    'use_edit_page_button': True,
    'show_toc_level': 2,
    'footer_start': ['copyright'],
    'footer_end': ['sphinx-version', 'theme-version'],
}
html_sidebars = {
    'index': [],
    'examples': [],
}


def setup(app):
    def release_mocked_cuda_cleanup(app, exception):
        # Autodoc mocks load no CUDA modules, and disappear before interpreter exit.
        if 'diffct.utils' in sys.modules:
            atexit.unregister(sys.modules['diffct.utils']._keep_cuda_modules_at_exit)

    app.connect('build-finished', release_mocked_cuda_cleanup)
