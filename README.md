<div align="center">
  <a href="https://github.com/andrewtavis/kwx"><img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/resources/kwx_logo_transparent.png" width=431 height=215></a>
</div>

--------------------------------------

[![rtd](https://img.shields.io/readthedocs/kwx.svg?logo=read-the-docs)](http://kwx.readthedocs.io/en/latest/)
[![ci](https://img.shields.io/github/workflow/status/andrewtavis/kwx/CI?logo=github)](https://github.com/andrewtavis/kwx/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/andrewtavis/kwx/branch/main/graphs/badge.svg)](https://codecov.io/gh/andrewtavis/kwx)
[![quality](https://img.shields.io/codacy/grade/25f1aa66a591458ab407f712c02af17c?logo=codacy)](https://app.codacy.com/gh/andrewtavis/kwx/dashboard)
[![pyversions](https://img.shields.io/pypi/pyversions/kwx.svg?logo=python&logoColor=FFD43B&color=306998)](https://pypi.org/project/kwx/)
[![pypi](https://img.shields.io/pypi/v/kwx.svg?color=4B8BBE)](https://pypi.org/project/kwx/)
[![pypistatus](https://img.shields.io/pypi/status/kwx.svg)](https://pypi.org/project/kwx/)
[![license](https://img.shields.io/github/license/andrewtavis/kwx.svg)](https://github.com/andrewtavis/kwx/blob/main/LICENSE.txt)
[![coc](https://img.shields.io/badge/coc-Contributor%20Covenant-ff69b4.svg)](https://github.com/andrewtavis/kwx/blob/main/.github/CODE_OF_CONDUCT.md)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![colab](https://img.shields.io/badge/%20-Open%20in%20Colab-097ABB.svg?logo=google-colab&color=097ABB&labelColor=525252)](https://colab.research.google.com/github/andrewtavis/kwx)

### BERT, LDA, and TFIDF based keyword extraction in Python

**kwx** is a toolkit for multilingual keyword extraction based on Google's [BERT](https://github.com/google-research/bert) and [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation). The package provides a suite of methods to process texts of any language to varying degrees and then extract and analyze keywords from the created corpus (see [kwx.languages](https://github.com/andrewtavis/kwx/blob/main/src/kwx/languages.py) for the various degrees of language support). A unique focus is allowing users to decide which words to not include in outputs, thereby guaranteeing sensible results that are in line with user intuitions.

For a thorough overview of the process and techniques see the [Google slides](https://docs.google.com/presentation/d/1BNddaeipNQG1mUTjBYmrdpGC6xlBvAi3rapT88fkdBU/edit?usp=sharing), and reference the [documentation](https://kwx.readthedocs.io/en/latest/) for explanations of the models and visualization methods.

# **Contents**<a id="contents"></a>
- [Installation](#installation)
- [Models](#models)
- [Usage](#usage)
- [Visuals](#visuals)
- [To-Do](#to-do)

# Installation [`↩`](#contents) <a id="installation"></a>

kwx can be downloaded from PyPI via pip or sourced directly from this repository:

```bash
pip install kwx
```

```bash
git clone https://github.com/andrewtavis/kwx.git
cd kwx
python setup.py install
```

```python
import kwx
```

# Models [`↩`](#contents) <a id="models"></a>

Implemented NLP modeling methods within [kwx.model](https://github.com/andrewtavis/kwx/blob/main/src/kwx/model.py) include:

### BERT

[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert) derives representations of words based on nlp models ran over open-source Wikipedia data. These representations are then leveraged to derive corpus topics.

kwx uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers) pretrained models. See their GitHub and [documentation](https://www.sbert.net/) for the available models.

### LDA

[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. In the case of kwx, documents or text entries are posited to be a mixture of a given number of topics, and the presence of each word in a text body comes from its relation to these derived topics.

Although not as computationally robust as some machine learning models, LDA provides quick results that are suitable for many applications. Specifically for keyword extraction, in most settings the results are similar to those of BERT in a fraction of the time.

### Other Methods

The user can also choose to simply query the most common words from a text corpus or compute TFIDF ([Term Frequency Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) keywords - those that are unique in a text body in comparison to another that's compared. The former method is used in kwx as a baseline to check model efficacy, and the latter is a useful baseline when a user has another text or text body to compare the target corpus against.

# Usage [`↩`](#contents) <a id="usage"></a>

Keyword extraction can be useful to analyze surveys, tweets and other kinds of social media posts, research papers, and further classes of texts. [examples/kw_extraction](https://github.com/andrewtavis/kwx/blob/main/examples/kw_extraction.ipynb) provides an example of how to use kwx by deriving keywords from tweets in the Kaggle [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) dataset.

The following outlines using kwx to derive keywords from a text corpus with `prompt_remove_words` as `True` (the user will be asked if some of the extracted words need to be replaced):

### Text Cleaning

```python
from kwx.utils import prepare_data

input_language = "english" # see kwx.languages for options
num_keywords = 15
num_topics = 10
ignore_words = ["words", "user", "knows", "they", "don't", "want"]

# kwx.utils.clean() can be used on a list of lists
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

### Keyword Extraction

```python
from kwx.model import extract_kws

# Remove n-grams for BERT training
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

Are there words that should be removed [y/n]? y
Type or copy word(s) to be removed: southwestair, united, virginamerica

The new BERT keywords are:

['late', 'baggage', 'service', 'flight', 'time', 'love', 'book', 'customer',
'response', 'hold', 'hour', 'cancel', 'cancelled_flighted', 'delay', 'plane']

Are there words that should be removed [y/n]? n
```

The model will be rerun until all words known to be unreasonable are removed for a suitable output. [kwx.model.gen_files](https://github.com/andrewtavis/kwx/blob/main/src/kwx/model.py) could also be used as a run-all function that produces a directory with a keyword text file and visuals (for experienced users wanting quick results).

# Visuals [`↩`](#contents) <a id="visuals"></a>

[kwx.visuals](https://github.com/andrewtavis/kwx/blob/main/src/kwx/visuals.py) includes the following functions for presenting and analyzing the results of keyword extraction:

### Topic Number Evaluation

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
  <img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/resources/gh_images/topic_num_eval.png" width="600" />
</p>

### t-SNE

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
  <img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/resources/gh_images/t_sne.png" width="600" />
</p>

### pyLDAvis

[pyLDAvis](https://github.com/bmabey/pyLDAvis) is included so that users can inspect LDA extracted topics, and further so that it can easily be generated for output files.

```python
from kwx.visuals import pyLDAvis_topics

pyLDAvis_topics(
    method="lda",
    text_corpus=text_corpus,
    num_topics=10,
    display_ipython=False,  # For Jupyter integration
)
```

<p align="middle">
  <img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/resources/gh_images/pyLDAvis.png" width="600" />
</p>

### Word Cloud

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
  <img src="https://raw.githubusercontent.com/andrewtavis/kwx/main/resources/gh_images/word_cloud.png" width="600" />
</p>

# To-Do [`↩`](#contents) <a id="to-do"></a>

Please see the [contribution guidelines](https://github.com/andrewtavis/kwx/blob/main/.github/CONTRIBUTING.md) if you are interested in contributing to this project. Work that is in progress or could be implemented includes:

- Including more methods to extract keywords [(see issue)](https://github.com/andrewtavis/kwx/issues/17)

- Adding key phrase extraction as an option for [kwx.model.extract_kws](https://github.com/andrewtavis/kwx/blob/main/src/kwx/model.py) [(see issues)](https://github.com/andrewtavis/kwx/issues/)

- Adding t-SNE and pyLDAvis style visualizations for BERT models

- Updates to [kwx.languages](https://github.com/andrewtavis/kwx/blob/main/src/kwx/languages.py) as lemmatization and other linguistic package dependencies evolve

- Creating, improving and sharing [examples](https://github.com/andrewtavis/kwx/tree/main/examples)

- Improving [tests](https://github.com/andrewtavis/kwx/tree/main/tests) for greater [code coverage](https://codecov.io/gh/andrewtavis/kwx)

- Updating and refining the [documentation](https://kwx.readthedocs.io/en/latest/)
