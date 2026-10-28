# Configuration file for the Sphinx documentation builder.

from pathlib import Path
import shutil
import sys
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / 'BaryonForge'
sys.path.insert(0, str(PROJECT_ROOT))

def run_apidoc(_):
    """Generate API pages from this checkout without copying external sources."""

    apidoc = shutil.which('sphinx-apidoc')
    command = ([apidoc] if apidoc else
               [sys.executable, '-m', 'sphinx.ext.apidoc'])
    subprocess.check_call(command + [
        '--force', '--ext-autodoc', '--ext-intersphinx', '-e',
        '-o', str(Path(__file__).resolve().parent),
        str(PACKAGE_ROOT), str(PACKAGE_ROOT / '*setup*'),
    ])


def setup(app):
    app.connect('builder-inited', run_apidoc)

# -- Project information

project = 'BaryonForge'
copyright = '2024, Anbajagane'
author = 'Dhayaa Anbajagane'

release = '1.0'
version = '1.0.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',  # optional: for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',  # optional: to include links to the source code
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

autodoc_member_order = "bysource"

templates_path = ['_templates']

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
