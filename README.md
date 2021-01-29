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

**Jump to:** [Methods](#methods) • [Algorithm](#algorithm) • [Examples](#examples) • [Usage](#usage) • [To-Do](#to-do)

**kwgen** is a toolkit for unsupervised keyword generation based on [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and Google's [BERT](https://github.com/google-research/bert). It provides a multilingual suite of methods to generate and analyze keywords from a corpus (group) of texts. See the [Google slides (WIP)](https://docs.google.com/presentation/d/1BNddaeipNQG1mUTjBYmrdpGC6xlBvAi3rapT88fkdBU/edit?usp=sharing) for a thorough overview of the process and techniques, and the [documentation](https://kwgen.readthedocs.io/en/latest/) for explanations of the models and visualization methods.

# Installation via PyPi
```bash
pip install kwgen
```

```python
import kwgen
```

# Methods

### LDA

[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. In the case of kwgen, documents or text entries are posited to be a mixture of a given number of topics, and the presence of each word in a text body comes from its relation to these derived topics.

### BERT

[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert) derives representations of words based running nlp models over open source Wikipedia data. These representations are then able to be leveraged to derive topics.

### LDA with BERT embeddings

The combination of LDA with BERT via an [kwgen.autoencoder](https://github.com/andrewtavis/kwgen/blob/main/kwgen/autoencoder.py).

### Other

The user can also choose to simply query the most common words from a text corpus or compute TFIDF (Term Frequency Inverse Document Frequency) words - those that are unique in a text body in comparison to another that's compared. The former method is used in kwgen as a baseline to check model efficacy, and the latter is a useful baseline when a user has another text or text body to compare the target against.

# Algorithm

The basic structure of kwgen's machine learning based keyword generation algorithms is the following:

- The user inputs a desired number of keywords
- The user inputs a number of topics to use, or this is determined by optimizing topic coherence and overlap across a potential topic numbers
- Topics are derived for the text corpus
- The prevalence of topics in the text corpus is found
  - For example: topic 1 is 25% coherent to the texts, topic 2 45%, and topic 3 30%
  - These percentages come from averaging topic coherence across all texts
- Words are selected from the derived topics based on their coherence to the text body
  - If a word has already been selected, then the next word in the topic will chosen
  - From the above example: the best 25%, 45% and 30% of words from topics 1-3 are selected
  - Words are selected from less coherent topics first (common words come from weakly coherent topics, and unique words come from those with strong coherence)
- The user is presented the generated keywords and asked if they're appropriate
  - They can then indicate words to be removed and replaced
  - Keywords are finalized when the user indicates that no more words need to be removed
- Optionally: the keywords are put into a text file, and this along with desired visuals is saved into a directory or zipped

# Usage

The following presents the usage of kwgen to derive keywords from a text corpus:

```python
import kwgen

```

# Examples

[examples.research_paper_kws](https://github.com/andrewtavis/kwgen/blob/main/examples/research_paper_kws.ipynb) provides an example of how to use kwgen by deriving keywords from research papers in the Kaggle [COVID-19 Open Research Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/).

# To-Do

- Including more methods to generate keywords
- Updates to [kwgen.languages](https://github.com/andrewtavis/kwgen/blob/main/kwgen/languages.py) as lemmatization and other linguistic package dependencies evolve
- Creating, improving and sharing [examples](https://github.com/andrewtavis/kwgen/tree/main/examples)
- Updating and refining the [documentation](https://kwgen.readthedocs.io/en/latest/)
