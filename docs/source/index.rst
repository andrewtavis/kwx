.. image:: https://raw.githubusercontent.com/andrewtavis/kwx/main/.github/resources/logo/kwx_logo_transparent.png
    :width: 431
    :height: 215
    :align: center
    :target: https://github.com/andrewtavis/kwx

|rtd| |ci_static_analysis| |ci_pytest| |pyversions| |pypi| |pypistatus| |license| |coc| |codestyle| |colab|

.. |rtd| image:: https://img.shields.io/readthedocs/kwx.svg?logo=read-the-docs
    :target: http://kwx.readthedocs.io/en/latest/

.. |ci_static_analysis| image:: https://img.shields.io/github/actions/workflow/status/andrewtavis/kwx/.github/workflows/ci_static_analysis.yaml?branch=main&label=ci&logo=ruff
    :target: https://github.com/andrewtavis/kwx/actions/workflows/ci_static_analysis.yaml

.. |ci_pytest| image:: https://img.shields.io/github/actions/workflow/status/andrewtavis/kwx/.github/workflows/ci_pytest.yaml?branch=main&label=build&logo=pytest
    :target: https://github.com/andrewtavis/kwx/actions/workflows/ci_pytest.yaml

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/kwx.svg?logo=python&logoColor=FFD43B&color=306998
    :target: https://pypi.org/project/kwx/

.. |pypi| image:: https://img.shields.io/pypi/v/kwx.svg?color=4B8BBE
    :target: https://pypi.org/project/kwx/

.. |pypistatus| image:: https://img.shields.io/pypi/status/kwx.svg
    :target: https://pypi.org/project/kwx/

.. |license| image:: https://img.shields.io/github/license/andrewtavis/kwx.svg
    :target: https://github.com/andrewtavis/kwx/blob/main/LICENSE.txt

.. |coc| image:: https://img.shields.io/badge/coc-Contributor%20Covenant-ff69b4.svg
    :target: https://github.com/andrewtavis/kwx/blob/main/.github/CODE_OF_CONDUCT.md

.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |colab| image:: https://img.shields.io/badge/%20-Open%20in%20Colab-097ABB.svg?logo=google-colab&color=097ABB&labelColor=525252
    :target: https://colab.research.google.com/github/andrewtavis/kwx

BERT, LDA, and TFIDF based keyword extraction in Python

Installation
------------

``kwx`` is available for installation via `uv <https://docs.astral.sh/uv/>`_ (recommended) or `pip <https://pypi.org/project/kwx/>`_.

.. code-block:: shell

    # Using uv (recommended - fast, Rust-based installer):
    uv pip install kwx

    # Or using pip:
    pip install kwx

.. code-block:: shell

    # For a development build of the package:
    git clone https://github.com/andrewtavis/kwx.git
    cd kwx

    # With uv (recommended):
    uv sync --all-extras  # install all dependencies
    source .venv/bin/activate  # activate venv (macOS/Linux)
    # .venv\Scripts\activate  # activate venv (Windows)

    # Or with pip:
    python -m venv .venv  # create virtual environment
    source .venv/bin/activate  # activate venv (macOS/Linux)
    # .venv\Scripts\activate  # activate venv (Windows)
    pip install -e .

.. code-block:: python

    import kwx

Contents
========

.. toctree::
    :maxdepth: 2

    model
    topic_model
    visuals
    languages
    utils

Development
===========

.. toctree::
    :maxdepth: 2

    notes

Project Indices
===============

* :ref:`genindex`
