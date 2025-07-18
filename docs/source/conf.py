# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DiffCT'
copyright = '2025, Yipeng Sun'
author = 'Yipeng Sun'
version = '1.1.7'
release = '1.1.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# HTML theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2c3e50',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = [
    'custom.css',
]

# HTML page configuration
html_title = f"{project} v{version} Documentation"
html_short_title = project
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Additional HTML context
html_context = {
    'display_github': True,
    'github_user': 'yipeng-sun',
    'github_repo': 'diffct',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

# SEO and meta tags
html_meta = {
    'description': 'DiffCT: Differentiable CT reconstruction operators for PyTorch',
    'keywords': 'CT, computed tomography, reconstruction, PyTorch, differentiable, medical imaging',
    'author': 'Yipeng Sun',
    'viewport': 'width=device-width, initial-scale=1.0',
}

# Search configuration
html_search_language = 'en'
html_search_options = {
    'type': 'default',
    'scorer': 'default',
}

# Auto-section label configuration for cross-references
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3
# Disable autosectionlabel to avoid duplicate labels
autosectionlabel_enabled = False

# Todo extension configuration
todo_include_todos = False

# Cross-reference configuration
nitpicky = True
nitpick_ignore = [
    ('py:class', 'torch.nn.Module'),
    ('py:class', 'torch.Tensor'),
    ('py:class', 'torch.autograd.Function'),
    ('py:meth', 'torch.autograd.Function.setup_context'),
    ('py:func', 'ctx.save_for_backward'),
    ('py:func', 'ctx.save_for_forward'),
    ('py:attr', 'ctx'),
    ('py:attr', 'ctx.needs_input_grad'),
]

# Additional cross-reference mappings
intersphinx_mapping.update({
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
})