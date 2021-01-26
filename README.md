<div align="center">
  <a href="https://github.com/andrewtavis/kwgen"><img src="https://github.com/andrewtavis/kwgen/blob/main/resources/kwgen_logo_transparent.png" width="517" height="223"></a>
</div>

--------------------------------------

[![rtd](https://img.shields.io/readthedocs/kwgen.svg?logo=read-the-docs)](http://kwgen.readthedocs.io/en/latest/)
[![travis](https://img.shields.io/travis/andrewtavis/kwgen.svg?logo=travis-ci)](https://travis-ci.org/andrewtavis/kwgen)
[![codecov](https://codecov.io/gh/andrewtavis/kwgen/branch/master/graphs/badge.svg)](https://codecov.io/gh/andrewtavis/kwgen)
[![pyversions](https://img.shields.io/pypi/pyversions/kwgen.svg?logo=python)](https://pypi.org/project/kwgen/)
[![pypi](https://img.shields.io/pypi/v/kwgen.svg)](https://pypi.org/project/kwgen/)
[![pypistatus](https://img.shields.io/pypi/status/kwgen.svg)](https://pypi.org/project/kwgen/)
[![license](https://img.shields.io/github/license/andrewtavis/kwgen.svg)](https://github.com/andrewtavis/kwgen/blob/main/LICENSE)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/andrewtavis/kwgen/blob/main/CONTRIBUTING.md)

### Unsupervised keyword generation in Python

**Jump to:** [Methods](#methods) • [Examples](#examples) • [To-Do](#to-do)

**kwgen** is a toolkit for unsupervised keyword generation based on [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and Google's [BERT](https://github.com/google-research/bert). It provides a multilingual suite of methods to generate and analyze keywords from a corpus of texts. See the [Google slides (WIP)](https://docs.google.com/presentation/d/1BNddaeipNQG1mUTjBYmrdpGC6xlBvAi3rapT88fkdBU/edit?usp=sharing) for a thorough overview of the process and techniques.

# Installation via PyPi
```bash
pip install kwgen
```

```python
import kwgen
```

# Methods

### LDA

### BERT

### LDA with BERT embeddings

# Examples

[examples.research_paper_kws](https://github.com/andrewtavis/kwgen/blob/main/examples/research_paper_kws.ipynb) provides an example of how to use kwgen by deriving keywords from research papers in the Kaggle [COVID-19 Open Research Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/).

# To-Do

- Including more methods to generate keywords
- Updates as lemmatization and other package dependencies evolve
- Creating, improving and sharing [examples](https://github.com/andrewtavis/kwgen/tree/main/examples)
