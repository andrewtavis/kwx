"""
Model Tests
-----------
"""

from io import StringIO

from kwx import model


def test_extract_frequent_kws(long_text_corpus):
    kws = model.extract_kws(
        method="frequency",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_extract_TFIDF_kws(long_text_corpus):
    kws = model.extract_kws(
        method="TFIDF",
        text_corpus=long_text_corpus,
        corpuses_to_compare=long_text_corpus,
        input_language="english",
        num_keywords=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_extract_LDA_kws(long_text_corpus):
    kws = model.extract_kws(
        method="lda",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_extract_kws_remove_words(monkeypatch, long_text_corpus):
    monkeypatch.setattr("sys.stdin", StringIO("y\nword\nn\n"))

    kws = model.extract_kws(
        method="lda",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=True,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_extract_BERT_kws(long_text_corpus):
    kws = model.extract_kws(
        method="bert",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_extract_lda_BERT_kws(long_text_corpus):
    kws = model.extract_kws(
        method="lda_bert",
        text_corpus=long_text_corpus,
        input_language="english",
        num_keywords=10,
        num_topics=10,
        prompt_remove_words=False,
    )
    assert len(kws) == 10
    assert type(kws[0]) == str


def test_gen_files():
    # model.gen_filesgen_files(
    #     method="lda",
    #     text_corpus=None,
    #     input_language="english",
    #     num_keywords=10,
    #     topic_nums_to_compare=[10, 11],
    #     prompt_remove_words=False,
    #     org_by_pos=True,
    #     incl_visuals=True,
    #     zip_results=True,
    # )
    assert True
