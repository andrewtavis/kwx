<div align="center">
  <a href="https://github.com/andrewtavis/kwx"><img src="https://github.com/andrewtavis/kwx/blob/main/resources/kwx_logo_transparent.png" width=60% height=60%></a>
</div>

--------------------------------------

[![rtd](https://img.shields.io/readthedocs/kwx.svg?logo=read-the-docs)](http://kwx.readthedocs.io/en/latest/)
[![travis](https://img.shields.io/travis/andrewtavis/kwx.svg?logo=travis-ci)](https://travis-ci.org/andrewtavis/kwx)
[![codecov](https://codecov.io/gh/andrewtavis/kwx/branch/master/graphs/badge.svg)](https://codecov.io/gh/andrewtavis/kwx)
[![pyversions](https://img.shields.io/pypi/pyversions/kwx.svg?logo=python)](https://pypi.org/project/kwx/)
[![pypi](https://img.shields.io/pypi/v/kwx.svg)](https://pypi.org/project/kwx/)
[![pypistatus](https://img.shields.io/pypi/status/kwx.svg)](https://pypi.org/project/kwx/)
[![license](https://img.shields.io/github/license/andrewtavis/kwx.svg)](https://github.com/andrewtavis/kwx/blob/main/LICENSE)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/andrewtavis/kwx/blob/main/CONTRIBUTING.md)

### Unsupervised keyword extraction in Python

**Jump to:** [Models](#models) • [Algorithm](#algorithm) • [Usage](#usage) • [Visuals](#visuals) • [To-Do](#to-do)

**kwx** is a toolkit for unsupervised keyword extraction based on [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and Google's [BERT](https://github.com/google-research/bert). It provides a multilingual suite of methods to process texts and then extract and analyze keywords from the created corpus. A unique focus is allowing users to decide which words to not include in outputs, thereby allowing them to use their own intuitions to fine tune the modeling process.

For a thorough overview of the process and techniques see the [Google slides](https://docs.google.com/presentation/d/1BNddaeipNQG1mUTjBYmrdpGC6xlBvAi3rapT88fkdBU/edit?usp=sharing), and reference the [documentation](https://kwx.readthedocs.io/en/latest/) for explanations of the models and visualization methods. Also see [kwx.languages](https://github.com/andrewtavis/kwx/blob/main/kwx/languages.py) for all available languages.

# Installation via PyPi
```bash
pip install kwx
```

```python
import kwx
```

# Models

### LDA

[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. In the case of kwx, documents or text entries are posited to be a mixture of a given number of topics, and the presence of each word in a text body comes from its relation to these derived topics.

Although not as statistically strong as the following models, LDA provides quick results that are suitable for many applications.

### BERT

[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert) derives representations of words based running nlp models over open source Wikipedia data. These representations are then able to be leveraged to derive corpus topics.

### LDA with BERT embeddings

The combination of LDA with BERT via [kwx.autoencoder](https://github.com/andrewtavis/kwx/blob/main/kwx/autoencoder.py).

### Other

The user can also choose to simply query the most common words from a text corpus or compute TFIDF ([Term Frequency Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) keywords - those that are unique in a text body in comparison to another that's compared. The former method is used in kwx as a baseline to check model efficacy, and the latter is a useful baseline when a user has another text or text body to compare the target corpus against.

# Algorithm

The structure `kwx.model.extract_kws`, kwx's natural language processing keyword extraction algorithm, is the following:

- The user inputs a desired number of keywords
- The user inputs a number of topics to use, or this is determined by optimizing topic coherence and overlap across potential topic numbers
- The texts are fully cleaned and tokenized ([see kwx.utils.clean_and_tokenize_texts](https://github.com/andrewtavis/kwx/blob/main/kwx/utils.py))
- Topics are derived for the text corpus
- The prevalence of topics in the text corpus is found
  - For example: topic 1 is 25% coherent to the texts, topic 2 45%, and topic 3 30%
  - These percentages come from averaging topic coherence across all texts
- Words are selected from the derived topics based on their coherence to the text body
  - If a word has already been selected, then the next word in the topic will be chosen
  - From the above example: the best 25%, 45% and 30% of words from topics 1-3 are selected
  - Words are selected from less coherent topics first (common words come from weakly coherent topics, and unique words come from those with strong coherence)
- The user is presented the extracted keywords and asked if they're appropriate
  - They can then indicate words to be removed and replaced
  - Keywords are finalized when the user indicates that no more words need to be replaced
- Optionally: the keywords are put into a text file, and this along with desired visuals is saved into a directory or zipped (see [kwx.model.gen_files](https://github.com/andrewtavis/kwx/blob/main/kwx/model.py))

# Usage

Keyword extraction can be useful to analyze surveys, tweets, other kinds of social media posts, research papers, and further classes of texts. [examples.kw_extraction](https://github.com/andrewtavis/kwx/blob/main/examples/kw_extraction.ipynb) provides an example of how to use kwx by deriving keywords from tweets in the Kaggle [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) dataset.

The following `pseudoscope` presents a brief outline of using kwx to derive keywords from a text corpus with `prompt_remove_words` as `True` (the user will be asked if some of the extracted words need to be replaced):

```python
from kwx.utils import prepare_data
from kwx.model import extract_kws

input_language = "english"
num_keywords = 10
num_topics = 10
ignore_words = ["words", "user", "knows", "they", "don't", "want"]

# Arguments from examples.kw_extraction
text_corpus = prepare_data(
    data='df-or-csv/xlsx-path',
    target_cols='cols-where-texts-are',
    input_language=input_language,
    min_freq=2,  # remove infrequent words
    min_word_len=4,  # remove small words
    sample_size=1,  # sample size for testing)
)[0]

bert_kws = extract_kws(
    method='BERT',
    text_corpus=text_corpus,
    input_language=input_language,
    output_language=None,  # allows the output to be translated
    num_keywords=num_keywords,
    num_topics=num_topics,
    corpuses_to_compare=None,  # for TFIDF
    return_topics=False,  # to inspect topics rather than produce kws
    ignore_words=ignore_words,
    prompt_remove_words=True,  # check words with user
)
```

The model will be re-ran until all words known to be unreasonable are removed for a suitable output. `kwx.model.gen_files` could also be used as a run-all function that produces a directory with a keyword text file and visuals (for experienced users wanting quick results).

# Visuals

[kwx.visuals](https://github.com/andrewtavis/kwx/blob/main/kwx/visuals.py) includes functions for both presenting and analyzing the results of keyword extraction.

### Topic Number Evaluation

A graph of topic coherence and overlap given a variable number of topics to derive keywords from.

```python
from kwx.visuals import graph_topic_num_evals

graph_topic_num_evals(
    method=["lda", "bert", "lda_bert"],
    text_corpus=text_corpus,
    input_language=input_language,
    num_keywords=num_keywords,
    topic_nums_to_compare=list(range(5,15)),
    metrics=True, # stability and coherence
    return_ideal_metrics=False, # selects ideal model given metrics for kwx.model.gen_files
)
```

### Word Cloud

Word clouds via [wordcloud](https://github.com/amueller/word_cloud) are included for a basic representation of the text corpus - specifically being a way to convey basic visual information to potential stakeholders. The following figure from [examples.kw_extraction](https://github.com/andrewtavis/kwx/blob/main/examples/kw_extraction.ipynb) shows a word cloud generated from tweets of US air carrier passengers:

```python
from kwx.visuals import gen_word_cloud

ignore_words = []

gen_word_cloud(
    text_corpus=text_corpus,
    input_language=input_language,
    ignore_words=None,
    height=500,
)
```

### pyLDAvis

[pyLDAvis](https://github.com/bmabey/pyLDAvis) is included so that users can inspect LDA extracted topics, and further so that it can easily be generated for output files.

```python
from kwx.visuals import pyLDAvis_topics

pyLDAvis_topics(
    method="lda",
    text_corpus=text_corpus,
    input_language=input_language,
    num_topics=10,
    display_ipython=False,  # For Jupyter integration
)
```

### t-SNE

[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) allows the user to visualize their topic distribution in both two and three dimensions. Currently available just for LDA, this technique provides another check for model suitability.

```python
from kwx.visuals import t_sne

t_sne(
    dimension="both",  # 2d and 3d are options
    text_corpus=text_corpus,
    num_topics=10,
    remove_3d_outliers=True,
)
```

# To-Do

- Including more methods to extract keywords, as well as improving the current ones
- Adding BERT [sentence-transformers](https://github.com/UKPLab/sentence-transformers) language models as an argument in [kwx.model.extract_kws](https://github.com/andrewtavis/kwx/blob/main/kwx/model.py)
- Allowing key phrase extraction
- Adding t-SNE and pyLDAvis style visualizations for BERT models
- Including more options to fine tune the cleaning process in [kwx.utils](https://github.com/andrewtavis/kwx/blob/main/kwx/utils.py)
- Updates to [kwx.languages](https://github.com/andrewtavis/kwx/blob/main/kwx/languages.py) as lemmatization and other linguistic package dependencies evolve
- Creating, improving and sharing [examples](https://github.com/andrewtavis/kwx/tree/main/examples)
- Updating and refining the [documentation](https://kwx.readthedocs.io/en/latest/)
