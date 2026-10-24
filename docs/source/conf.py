# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import atexit
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('_ext'))

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
    'sphinx_llms_txt',
    'search_chunks',
]

llms_txt_title = 'diffct'
llms_txt_summary = (
    'Differentiable CUDA projectors and matched adjoints for computed tomography in PyTorch. '
    'Parallel, fan, cone and helical geometries, arbitrary trajectories, curved detectors, multi-GPU.'
)

templates_path = ['_templates']
exclude_patterns = []
autodoc_typehints = 'description'
autosummary_generate = True
copybutton_prompt_text = r'\$ |>>> '
copybutton_prompt_is_regexp = True

html_theme = 'pydata_sphinx_theme'
html_title = f'diffct {release}'
html_baseurl = 'https://sypsyp97.github.io/diffct/'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_favicon = '_static/favicon.svg'
html_logo = '_static/logo.svg'
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
    # Generated API stubs are not in the repository, so they get no edit link.
    'secondary_sidebar_items': {'*': ['page-toc', 'edit-this-page'], 'generated/*': ['page-toc']},
    'use_edit_page_button': True,
    'show_toc_level': 2,
    'footer_start': ['copyright', 'llms-links'],
    'footer_end': ['sphinx-version', 'theme-version'],
}
html_sidebars = {
    'index': [],
    'examples': [],
    'getting_started': [],
    'changelog': [],
}


def setup(app):
    def release_mocked_cuda_cleanup(app, exception):
        # Autodoc mocks load no CUDA modules, and disappear before interpreter exit.
        if 'diffct.utils' in sys.modules:
            atexit.unregister(sys.modules['diffct.utils']._keep_cuda_modules_at_exit)

    app.connect('build-finished', release_mocked_cuda_cleanup)

    def add_semantic_search(app, pagename, templatename, context, doctree):
        if pagename == 'search':
            app.add_js_file('semantic-search.js', loading_method='defer')

    app.connect('html-page-context', add_semantic_search)
