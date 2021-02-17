"""
Model Tests
-----------
"""

from kwx import model


def test_get_topic_words():
    assert True


def test_get_coherence():
    assert True


def test__order_and_subset_by_coherence():
    assert True


def test_extract_frequent_kws(text_corpus):
    kws = model.extract_kws(
        method="frequency",
        text_corpus=text_corpus,
        input_language="english",
        num_keywords=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_extract_TFIDF_kws(text_corpus):
    kws = model.extract_kws(
        method="TFIDF",
        text_corpus=text_corpus,
        corpuses_to_compare=text_corpus,
        input_language="english",
        num_keywords=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_extract_LDA_kws(text_corpus):
    kws = model.extract_kws(
        method="lda",
        text_corpus=text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_extract_BERT_kws(text_corpus):
    kws = model.extract_kws(
        method="bert",
        text_corpus=text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_extract_lda_BERT_kws(text_corpus):
    kws = model.extract_kws(
        method="lda_bert",
        text_corpus=text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_gen_files():
    assert True
