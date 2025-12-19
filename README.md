<div align="center">
  <a href="https://github.com/andrewtavis/kwx"><img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/.github/resources/logo/kwx_logo_transparent.png" width=431 height=215></a>
</div>

<ol></ol>

[![rtd](https://img.shields.io/readthedocs/kwx.svg?logo=read-the-docs)](http://kwx.readthedocs.io/en/latest/)
[![pr_ci](https://img.shields.io/github/actions/workflow/status/andrewtavis/kwx/.github/workflows/pr_ci.yaml?branch=main&label=ci&logo=ruff)](https://github.com/andrewtavis/kwx/actions/workflows/pr_ci.yaml)
[![python_package_ci](https://img.shields.io/github/actions/workflow/status/andrewtavis/kwx/.github/workflows/python_package_ci.yaml?branch=main&label=build&logo=pytest)](https://github.com/andrewtavis/kwx/actions/workflows/python_package_ci.yaml)
[![codecov](https://codecov.io/gh/andrewtavis/kwx/branch/main/graphs/badge.svg)](https://codecov.io/gh/andrewtavis/kwx)
[![pyversions](https://img.shields.io/pypi/pyversions/kwx.svg?logo=python&logoColor=FFD43B&color=306998)](https://pypi.org/project/kwx/)
[![pypi](https://img.shields.io/pypi/v/kwx.svg?color=4B8BBE)](https://pypi.org/project/kwx/)
[![pypistatus](https://img.shields.io/pypi/status/kwx.svg)](https://pypi.org/project/kwx/)
[![license](https://img.shields.io/github/license/andrewtavis/kwx.svg)](https://github.com/andrewtavis/kwx/blob/main/LICENSE.txt)
[![coc](https://img.shields.io/badge/coc-Contributor%20Covenant-ff69b4.svg)](https://github.com/andrewtavis/kwx/blob/main/.github/CODE_OF_CONDUCT.md)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![colab](https://img.shields.io/badge/%20-Open%20in%20Colab-097ABB.svg?logo=google-colab&color=097ABB&labelColor=525252)](https://colab.research.google.com/github/andrewtavis/kwx)

## BERT, LDA, and TFIDF based keyword extraction in Python

**kwx** is a toolkit for multilingual keyword extraction based on Google's [BERT](https://github.com/google-research/bert), [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and [Term Frequency Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). The package provides a suite of methods to process texts of any language to varying degrees and then extract and analyze keywords from the created corpus (see [kwx.languages](https://github.com/andrewtavis/kwx/blob/main/src/kwx/languages.py) for the various degrees of language support). A unique focus is allowing users to decide which words to not include in outputs, thereby guaranteeing sensible results that are in line with user intuitions.

For a thorough overview of the process and techniques see the [Google slides](https://docs.google.com/presentation/d/1BNddaeipNQG1mUTjBYmrdpGC6xlBvAi3rapT88fkdBU/edit?usp=sharing), and reference the [documentation](https://kwx.readthedocs.io/en/latest/) for explanations of the models and visualization methods.

<a id="contents"></a>

## **Contents**

- [Installation](#installation-)
- [Models](#models-)
  - [BERT](#bert-)
  - [LDA](#lda-)
  - [TFIDF](#tfidf-)
  - [Word Frequency](#word-frequency-)
- [Usage](#usage-)
  - [Text Cleaning](#text-cleaning-)
  - [Keyword Extraction](#keyword-extraction-)
- [Visuals](#visuals-)
  - [Topic Number Evaluation](#topic-number-evaluation-)
  - [t-SNE](#t-sne-)
  - [pyLDAvis](#pyldavis-)
  - [Word Cloud](#word-cloud-)
- [Development environment](#development-environment-)
- [To-Do](#to-do-)

<a id="installation"></a>

## Installation [`⇧`](#contents)

kwx can be downloaded from PyPI via pip or sourced directly from this repository:

```bash
pip install kwx
```

```bash
# For a development build of the package:
git clone https://github.com/andrewtavis/kwx.git
cd kwx
python setup.py install
```

```python
import kwx
```

<a id="models"></a>

## Models [`⇧`](#contents)

Implemented NLP modeling methods within [kwx.model](https://github.com/andrewtavis/kwx/blob/main/src/kwx/model.py) include:

<a id="bert"></a>

### BERT [`⇧`](#contents)

[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert) derives representations of words based on nlp models ran over open-source Wikipedia data. These representations are then leveraged to derive corpus topics.

kwx uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers) pretrained models. See their GitHub and [documentation](https://www.sbert.net/) for the available models.

<a id="lda"></a>

### LDA [`⇧`](#contents)

[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. In the case of kwx, documents or text entries are posited to be a mixture of a given number of topics, and the presence of each word in a text body comes from its relation to these derived topics.

Although not as computationally robust as some machine learning models, LDA provides quick results that are suitable for many applications. Specifically for keyword extraction, in most settings the results are similar to those of BERT in a fraction of the time.

<a id="tfidf"></a>

### TFIDF [`⇧`](#contents)

The user can also compute [Term Frequency Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) keywords - those that are unique in a text body in comparison to another that's compared. This is a useful baseline when a user has another text or text body to compare the target corpus against.

<a id="word-frequency"></a>

### Word Frequency [`⇧`](#contents)

Finally a user can simply query the most common words from a text corpus. This method is used in kwx as a baseline to check model efficacy.

<a id="usage"></a>

## Usage [`⇧`](#contents)

Keyword extraction can be useful to analyze surveys, tweets and other kinds of social media posts, research papers, and further classes of texts. [examples/kw_extraction](https://github.com/andrewtavis/kwx/blob/main/examples/kw_extraction.ipynb) provides an example of how to use kwx by deriving keywords from tweets in the Kaggle [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) dataset.

The following outlines using kwx to derive keywords from a text corpus with `prompt_remove_words` as `True` (the user will be asked if some of the extracted words need to be replaced):

<a id="text-cleaning"></a>

### Text Cleaning [`⇧`](#contents)

```python
from kwx.utils import prepare_data

input_language = "english" # see kwx.languages for options

# kwx.utils.clean() can be used on a list of lists.
text_corpus = prepare_data(
    data="df_or_csv_xlsx_path",
    target_cols="cols_where_texts_are",
    input_language=input_language,
    min_token_freq=0,  # for BERT
    min_token_len=0,  # for BERT
    remove_stopwords=False,  # for BERT
    verbose=True,
)
```

<a id="keyword-extraction"></a>

### Keyword Extraction [`⇧`](#contents)

```python
from kwx.model import extract_kws

num_keywords = 15
num_topics = 10
ignore_words = ["words", "user", "knows", "they", "don't", "want"]

# Remove n-grams for BERT training.
corpus_no_ngrams = [
    " ".join([t for t in text.split(" ") if "_" not in t]) for text in text_corpus
]

# We can pass keywords for sentence_transformers.SentenceTransformer.encode,
# gensim.models.ldamulticore.LdaMulticore, or sklearn.feature_extraction.text.TfidfVectorizer
bert_kws = extract_kws(
    method="BERT", # "BERT", "LDA", "TFIDF", "frequency"
    bert_st_model="xlm-r-bert-base-nli-stsb-mean-tokens",
    text_corpus=corpus_no_ngrams,  # automatically tokenized if using LDA
    input_language=input_language,
    output_language=None,  # allows the output to be translated
    num_keywords=num_keywords,
    num_topics=num_topics,
    corpuses_to_compare=None,  # for TFIDF
    ignore_words=ignore_words,
    prompt_remove_words=True,  # check words with user
    show_progress_bar=True,
    batch_size=32,
)
```

```_output
The BERT keywords are:

['time', 'flight', 'plane', 'southwestair', 'ticket', 'cancel', 'united', 'baggage',
'love', 'virginamerica', 'service', 'customer', 'delay', 'late', 'hour']

Should words be removed [y/n]? y
Type or copy word(s) to be removed: southwestair, united, virginamerica

The new BERT keywords are:

['late', 'baggage', 'service', 'flight', 'time', 'love', 'book', 'customer',
'response', 'hold', 'hour', 'cancel', 'cancelled_flighted', 'delay', 'plane']

Should words be removed [y/n]? n
```

The model will be rerun until all words known to be unreasonable are removed for a suitable output. [kwx.model.gen_files](https://github.com/andrewtavis/kwx/blob/main/src/kwx/model.py) could also be used as a run-all function that produces a directory with a keyword text file and visuals (for experienced users wanting quick results).

<a id="visuals"></a>

## Visuals [`⇧`](#contents)

[kwx.visuals](https://github.com/andrewtavis/kwx/blob/main/src/kwx/visuals.py) includes the following functions for presenting and analyzing the results of keyword extraction:

<a id="topic-number-evaluation"></a>

### Topic Number Evaluation [`⇧`](#contents)

A graph of topic coherence and overlap given a variable number of topics to derive keywords from.

```python
from kwx.visuals import graph_topic_num_evals
import matplotlib.pyplot as plt

graph_topic_num_evals(
    method=["lda", "bert"],
    text_corpus=text_corpus,
    num_keywords=num_keywords,
    topic_nums_to_compare=list(range(5, 15)),
    metrics=True, #  stability and coherence
)
plt.show()
```

<p align="middle">
  <img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/.github/resources/images/topic_num_eval.png" width="600" />
</p>

<a id="t-sne"></a>

### t-SNE [`⇧`](#contents)

[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) allows the user to visualize their topic distribution in both two and three dimensions. Currently available just for LDA, this technique provides another check for model suitability.

```python
from kwx.visuals import t_sne
import matplotlib.pyplot as plt

t_sne(
    dimension="both",  # 2d and 3d are options
    text_corpus=text_corpus,
    num_topics=10,
    remove_3d_outliers=True,
)
plt.show()
```

<p align="middle">
  <img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/.github/resources/images/t_sne.png" width="600" />
</p>

<a id="pyldavis"></a>

### pyLDAvis [`⇧`](#contents)

[pyLDAvis](https://github.com/bmabey/pyLDAvis) is included so that users can inspect LDA extracted topics, and further so that it can easily be generated for output files.

```python
from kwx.visuals import pyLDAvis_topics

pyLDAvis_topics(
    method="lda",
    text_corpus=text_corpus,
    num_topics=10,
    display_ipython=False,  # for Jupyter integration
)
```

<p align="middle">
  <img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/.github/resources/images/pyLDAvis.png" width="600" />
</p>

<a id="word-cloud"></a>

### Word Cloud [`⇧`](#contents)

Word clouds via [wordcloud](https://github.com/amueller/word_cloud) are included for a basic representation of the text corpus - specifically being a way to convey basic visual information to potential stakeholders. The following figure from [examples/kw_extraction](https://github.com/andrewtavis/kwx/blob/main/examples/kw_extraction.ipynb) shows a word cloud generated from tweets of US air carrier passengers:

```python
from kwx.visuals import gen_word_cloud

ignore_words = ["words", "user", "knows", "they", "don't", "want"]

gen_word_cloud(
    text_corpus=text_corpus,
    ignore_words=None,
    height=500,
)
```

<p align="middle">
  <img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/.github/resources/images/word_cloud.png" width="600" />
</p>

<a name="development-environment-"></a>

## Development environment [`⇧`](#contents)

Please follow the steps below to set up your development environment for kwx contributions.

### Clone repository

```bash
# Clone your fork of the repo into the current directory.
git clone https://github.com/<your-username>/kwx.git
# Navigate to the newly cloned directory.
cd kwx
# Assign the original repo to a remote called "upstream".
git remote add upstream https://github.com/andrewtavis/kwx.git
```

- Now, if you run `git remote -v` you should see two remote repositories named:
  - `origin` (forked repository)
  - `upstream` (kwx repository)

### Conda environment

Download [Anaconda](https://www.anaconda.com/download) if you don't have it installed already.

```bash
conda env create --file environment.yaml
conda activate kwx-dev
```

### pip environment

Create a virtual environment, activate it and install dependencies:

```bash
# Unix or MacOS:
python3 -m venv venv
source venv/bin/activate

# Windows:
python -m venv venv
venv\Scripts\activate.bat

# After activating venv:
pip install --upgrade pip
pip install -r requirements-dev.txt

# To install the CLI for local development:
pip install -e .
```

### pre-commit

Install [pre-commit](https://pre-commit.com/) to ensure that each of your commits is properly checked against our linter and formatters:

```bash
# In the project root:
pre-commit install

# Then test the pre-commit hooks to see how it works:
pre-commit run --all-files
```

> [!NOTE]
> pre-commit is Python package that can be installed via pip or any other Python package manager. You can also find it in our [requirements-dev.txt](./requirements-dev.txt) file.
>
> ```bash
> pip install pre-commit
> ```

> [!NOTE]
> If you are having issues with pre-commit and want to send along your changes regardless, you can ignore the pre-commit hooks via the following:
>
> ```bash
> git commit --no-verify -m "COMMIT_MESSAGE"
> ```

<a id="to-do"></a>

## To-Do [`⇧`](#contents)

Please see the [contribution guidelines](https://github.com/andrewtavis/kwx/blob/main/.github/CONTRIBUTING.md) if you are interested in contributing to this project. Work that is in progress or could be implemented includes:

- Including more methods to extract keywords [(see issue)](https://github.com/andrewtavis/kwx/issues/17)

- Adding key phrase extraction as an option for [kwx.model.extract_kws](https://github.com/andrewtavis/kwx/blob/main/src/kwx/model.py) [(see issues)](https://github.com/andrewtavis/kwx/issues/)

- Adding t-SNE and pyLDAvis style visualizations for BERT models [(see issues\)](https://github.com/andrewtavis/kwx/issues/45)

- Converting the translation feature over to use another translation api rather than [py-googletrans](https://github.com/ssut/py-googletrans) [(see issue)](https://github.com/andrewtavis/kwx/issues/44)

- Updates to [kwx.languages](https://github.com/andrewtavis/kwx/blob/main/src/kwx/languages.py) as lemmatization and other linguistic package dependencies evolve

- Creating, improving and sharing [examples](https://github.com/andrewtavis/kwx/tree/main/examples)

- Improving [tests](https://github.com/andrewtavis/kwx/tree/main/tests) for greater [code coverage](https://codecov.io/gh/andrewtavis/kwx)

- Updating and refining the [documentation](https://kwx.readthedocs.io/en/latest/)
