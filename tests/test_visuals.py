"""
Visuals Tests
-------------
"""

import os
from pytest_cov.embed import cleanup_on_sigterm

import matplotlib.pyplot as plt

from kwx import visuals

os.environ["TOKENIZERS_PARALLELISM"] = "false"
cleanup_on_sigterm()


def test_graph_topic_num_evals(monkeypatch, long_text_corpus):
    monkeypatch.setattr(plt, "show", lambda: None)
    visuals.graph_topic_num_evals(
        method=["lda", "lda_bert", "bert"],
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        topic_nums_to_compare=[9, 10],
        save_file=False,
        return_ideal_metrics=False,
        verbose=False,
    )


def test_return_ideal_metrics(long_text_corpus):
    assert (
        type(
            visuals.graph_topic_num_evals(
                method=["lda"],
                text_corpus=long_text_corpus,
                input_language="english",
                num_keywords=10,
                topic_nums_to_compare=[9, 10],
                save_file=False,
                return_ideal_metrics=True,
                verbose=False,
            )[1]
        )
        == int
    )


def test_gen_word_cloud(monkeypatch, long_text_corpus):
    monkeypatch.setattr(plt, "show", lambda: None)
    visuals.gen_word_cloud(
        text_corpus=long_text_corpus,
        input_language="english",
        ignore_words="word",
        save_file=False,
    )


def test_gen_word_cloud_zip(monkeypatch, long_text_corpus):
    monkeypatch.setattr(plt, "show", lambda: None)
    visuals.gen_word_cloud(
        text_corpus=long_text_corpus,
        input_language="english",
        ignore_words="word",
        save_file="tests/test.zip",
    )
    os.remove("tests/test.zip")


def test_pyLDAvis_topics(long_text_corpus):
    visuals.pyLDAvis_topics(
        method="lda",
        text_corpus=long_text_corpus,
        input_language="english",
        num_topics=10,
        min_freq=2,
        min_word_len=3,
        sample_size=1,
        save_file="tests",
        display_ipython=False,
    )

    os.remove("tests/lda_topics.html")


def test_pyLDAvis_topics_zip(long_text_corpus):
    visuals.pyLDAvis_topics(
        method="lda",
        text_corpus=long_text_corpus,
        input_language="english",
        num_topics=10,
        min_freq=2,
        min_word_len=3,
        sample_size=1,
        save_file="tests/test.zip",
        display_ipython=False,
    )

    os.remove("tests/test.zip")


def test_t_sne(monkeypatch, long_text_corpus):
    monkeypatch.setattr(plt, "show", lambda: None)
    visuals.t_sne(
        dimension="both",
        text_corpus=long_text_corpus,
        num_topics=3,
        remove_3d_outliers=True,
        save_file=False,
    )

    visuals.t_sne(
        dimension="2d", text_corpus=long_text_corpus, num_topics=3, save_file=False,
    )

    visuals.t_sne(
        dimension="3d",
        text_corpus=long_text_corpus,
        num_topics=3,
        remove_3d_outliers=True,
        save_file=False,
    )
