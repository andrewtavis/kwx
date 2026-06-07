# SPDX-License-Identifier: BSD-3-Clause
"""
Model Tests
-----------
"""

import os
from io import StringIO

import gensim
import numpy as np
import pytest

from kwx import model

np.random.seed(42)


@pytest.mark.asyncio
async def test_extract_frequent_kws(long_text_corpus):
    kws = await model.extract_kws(
        method="frequency",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        prompt_remove_words=False,
    )
    required = [
        "virginamerica",
        "flight",
        "tco",
        "carrieunderwood",
        "ladygaga",
        "virginamerica_ladygaga",
        "virginamerica_ladygaga_carrieunderwood",
        "flights",
        "lax",
        "time",
    ]

    missing = [x for x in required if x not in kws]
    assert not missing, f"Missing keywords: {missing}"


@pytest.mark.asyncio
async def test_translate_kw_output(long_text_corpus):
    kws = await model.extract_kws(
        method="frequency",
        text_corpus=long_text_corpus,
        input_language="english",
        output_language="german",
        num_keywords=10,
        prompt_remove_words=False,
    )
    required = [
        "Virginiaamerika",
        "Flug",
        "tco",
        "carrieunderwood",
        "Ladygaga",
        "virginamerica_ladygaga",
        "virginamerica_ladygaga_carrieunderwood",
        "Flüge",
        "lax",
        "Zeit",
    ]
    missing = [x for x in required if x not in kws]
    assert not missing, f"Missing keywords: {missing}"


@pytest.mark.asyncio
async def test_extract_TFIDF_kws(long_text_corpus):
    kws = await model.extract_kws(
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
        "virginamerica_ladygaga_carrieunderwood",
        "virginamerica_ladygaga",
        "flights",
        "time",
        "lax",
    ]


if float(gensim.__version__[0]) >= 4.0:

    @pytest.mark.asyncio
    async def test_extract_LDA_kws(long_text_corpus):
        kws = await model.extract_kws(
            method="lda",
            text_corpus=long_text_corpus,
            input_language="english",
            num_keywords=10,
            num_topics=10,
            prompt_remove_words=False,
        )
        print(kws)
        assert kws == [
            "virginamerica",
            "tco",
            "lax",
            "flight",
            "carrieunderwood",
            "time",
            "trip",
            "booked",
            "virginamerica_ladygaga_carrieunderwood",
            "experience",
        ]

    @pytest.mark.asyncio
    async def test_extract_kws_remove_words(monkeypatch, long_text_corpus):
        monkeypatch.setattr("sys.stdin", StringIO("y\nvirginamerica\nn\n"))

        kws = await model.extract_kws(
            method="lda",
            text_corpus=long_text_corpus,
            input_language="english",
            num_keywords=10,
            num_topics=10,
            prompt_remove_words=True,
        )
        assert kws == [
            "flight",
            "tco",
            "ladygaga",
            "lax",
            "seat",
            "flew",
            "service",
            "carrieunderwood",
            "elevate",
            "virginamerica_ladygaga",
        ]


else:

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
            "virginamerica",
            "tco",
            "lax",
            "flight",
            "carrieunderwood",
            "time",
            "trip",
            "booked",
            "virginamerica_ladygaga_carrieunderwood",
            "experience",
        ]


@pytest.mark.asyncio
async def test_extract_BERT_kws(long_text_corpus):
    kws = await model.extract_kws(
        method="bert",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=False,
    )
    assert kws == [
        "virginamerica",
        "time",
        "tco",
        "flight",
        "elevate",
        "lax",
        "fly",
        "carrieunderwood",
        "service",
        "experience",
    ]


@pytest.mark.asyncio
async def test_gen_files(monkeypatch, long_text_corpus):
    monkeypatch.setattr("sys.stdin", StringIO("y\nrandom_word\nn\n"))

    await model.gen_files(
        method="lda",
        text_corpus=long_text_corpus,
        input_language="en_core_web_sm",
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
