"""
Model Tests
-----------
"""

import os
from io import StringIO

import numpy as np
from kwx import model

np.random.seed(42)


def test_extract_frequent_kws(long_text_corpus):
    kws = model.extract_kws(
        method="frequency",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        prompt_remove_words=False,
    )
    assert kws == [
        "virginamerica",
        "flight",
        "tco",
        "carrieunderwood",
        "ladygaga",
        "fly",
        "virginamerica_ladygaga",
        "virginamerica_ladygaga_carrieunderwood",
        "seat",
        "lax",
    ]


def test_translate_kw_output(long_text_corpus):
    kws = model.extract_kws(
        method="frequency",
        text_corpus=long_text_corpus,
        input_language="english",
        output_language="german",
        num_keywords=10,
        prompt_remove_words=False,
    )
    assert kws == [
        "VirginAmerica.",
        "Flug",
        "tco.",
        "Carrie underwood",
        "Lady Gaga",
        "Fliege",
        "virginamerica_ladygaga.",
        "virginamerica_ladygaga_carrieunderwood.",
        "Sitz",
        "lax",
    ]


def test_extract_TFIDF_kws(long_text_corpus):
    kws = model.extract_kws(
        method="TFIDF",
        text_corpus=long_text_corpus,
        corpuses_to_compare=long_text_corpus,
        input_language="english",
        num_keywords=10,
        prompt_remove_words=False,
    )
    assert kws == [
        "virginamerica",
        "flight",
        "tco",
        "carrieunderwood",
        "ladygaga",
        "fly",
        "virginamerica_ladygaga_carrieunderwood",
        "virginamerica_ladygaga",
        "seat",
        "time",
    ]


def test_extract_LDA_kws(long_text_corpus):
    kws = model.extract_kws(
        method="lda",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=False,
    )
    assert kws == [
        "virginamerica",
        "customer",
        "flight",
        "tco",
        "airline",
        "trip",
        "fly",
        "carrieunderwood",
        "bag",
        "week",
    ]


def test_extract_kws_remove_words(monkeypatch, long_text_corpus):
    monkeypatch.setattr("sys.stdin", StringIO("y\nvirginamerica\nn\n"))

    kws = model.extract_kws(
        method="lda",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=True,
    )
    assert kws == [
        "flight",
        "time",
        "seat",
        "traveler",
        "lax",
        "fly",
        "ladygaga",
        "virginamerica_ladygaga",
        "carrieunderwood",
        "virginamerica_ladygaga_carrieunderwood",
    ]


def test_extract_BERT_kws(long_text_corpus):
    kws = model.extract_kws(
        method="bert",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=False,
    )
    assert kws == [
        "virginamerica",
        "bad",
        "hawaii",
        "seat",
        "flight",
        "carrieunderwood",
        "fly",
        "tco",
        "time",
        "account",
    ]


def test_gen_files(monkeypatch, long_text_corpus):
    monkeypatch.setattr("sys.stdin", StringIO("y\nrandom_word\nn\n"))

    model.gen_files(
        method="lda",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        topic_nums_to_compare=[10, 11],
        prompt_remove_words=True,
        verbose=False,
        incl_most_freq=True,
        org_by_pos=True,
        incl_visuals=True,
        save_dir="text_corpus_kws.zip",
        zip_results=True,
    )

    os.remove("text_corpus_kws.zip")
