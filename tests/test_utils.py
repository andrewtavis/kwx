"""
Utilities Tests
---------------
"""

import os
import spacy
from io import StringIO

from kwx import utils


def test_load_data(df_texts):
    assert utils.load_data(data=df_texts, target_cols="text").equals(df_texts)


def test__combine_texts_to_str():
    texts = ["words", "to", "not", "be", "combined"]
    texts_lol = [["words", "to"], ["not"], ["be", "combined"]]
    ignore_words = ["not"]
    result_ignore_not = "words to be combined"
    result_not = "words to not be combined"

    assert (
        utils._combine_texts_to_str(text_corpus=texts, ignore_words=ignore_words)
        == result_ignore_not
    )

    assert (
        utils._combine_texts_to_str(text_corpus=texts_lol, ignore_words=ignore_words)
        == result_ignore_not
    )

    assert (
        utils._combine_texts_to_str(text_corpus=texts, ignore_words=None) == result_not
    )

    assert (
        utils._combine_texts_to_str(text_corpus=texts_lol, ignore_words=None)
        == result_not
    )


def test__lemmatize():
    try:
        nlp = spacy.load("en")
    except:
        os.system("python -m spacy download {}".format("en"))
        nlp = spacy.load("en")
    assert utils._lemmatize([["better"], ["walking"], ["dogs"]], nlp=nlp) == [
        ["well"],
        ["walk"],
        ["dog"],
    ]


def test_clean(list_texts):
    result = [
        "virginamerica sfo",
        "virginamerica",
        "virginamerica fly sfo seat",
        "fly virginamerica",
        "virginamerica fly",
        "virginamerica seat",
        "virginamerica love",
        "virginamerica love",
        "virginamerica",
        "virginamerica seat seat seat",
    ]
    assert (
        utils.clean(
            texts=list_texts,
            input_language="english",
            min_token_freq=2,
            min_token_len=3,
            min_tokens=0,
            max_token_index=-1,
            min_ngram_count=3,
            remove_stopwords=True,
            ignore_words=None,
            sample_size=1,
            verbose=False,
        )
        == result
    )

    result_no_small_words = [
        "virginamerica",
        "virginamerica",
        "virginamerica seat",
        "virginamerica",
        "virginamerica",
        "virginamerica seat",
        "virginamerica love",
        "virginamerica love",
        "virginamerica",
        "virginamerica seat seat seat",
    ]

    assert (
        utils.clean(
            texts=list_texts,
            input_language="english",
            min_token_freq=2,
            min_token_len=4,
            min_tokens=0,
            max_token_index=-1,
            min_ngram_count=3,
            remove_stopwords=True,
            ignore_words=None,
            sample_size=1,
            verbose=False,
        )
        == result_no_small_words
    )

    result_min_3_freq = [
        "virginamerica",
        "virginamerica",
        "virginamerica fly seat",
        "fly virginamerica",
        "virginamerica fly",
        "virginamerica seat",
        "virginamerica",
        "virginamerica",
        "virginamerica",
        "virginamerica seat seat seat",
    ]

    assert (
        utils.clean(
            texts=list_texts,
            input_language="english",
            min_token_freq=3,
            min_token_len=3,
            min_tokens=0,
            max_token_index=-1,
            min_ngram_count=3,
            remove_stopwords=True,
            ignore_words=None,
            sample_size=1,
            verbose=False,
        )
        == result_min_3_freq
    )

    assert (
        len(
            utils.clean(
                texts=list_texts,
                input_language="english",
                min_token_freq=3,
                min_token_len=3,
                min_tokens=0,
                max_token_index=-1,
                min_ngram_count=3,
                remove_stopwords=True,
                ignore_words=None,
                sample_size=0.8,
                verbose=False,
            )
        )
        == 8
    )


def test_prepare_data(df_texts):
    result = [
        "virginamerica sfo",
        "virginamerica",
        "virginamerica fly sfo seat",
        "fly virginamerica",
        "virginamerica fly",
        "virginamerica seat",
        "virginamerica love",
        "virginamerica love",
        "virginamerica",
        "virginamerica seat seat seat",
    ]
    assert (
        utils.prepare_data(
            data=df_texts,
            target_cols="text",
            input_language="english",
            min_token_freq=2,
            min_token_len=3,
            min_tokens=0,
            max_token_index=-1,
            min_ngram_count=3,
            remove_stopwords=True,
            ignore_words=None,
            sample_size=1,
            verbose=False,
        )
        == result
    )


def test_translate_output():
    assert utils.translate_output(
        outputs=["good"], input_language="english", output_language="german"
    ) == ["gut"]


def test_organize_by_pos():
    test_list = ["run", "jump", "dog", "tall"]
    assert str(utils.organize_by_pos(test_list, output_language="english")) == str(
        "{'Nouns:': [dog], 'Adjectives:': [tall], 'Verbs:': [run, jump]}"
    )


def test_prompt_for_word_removal(monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO("y\nalso\nn\n"))

    assert utils.prompt_for_word_removal(words_to_ignore="ignore")[0] == [
        "ignore",
        "also",
    ]


def test__prepare_corpus_path(short_text_corpus, df_texts):
    df_texts.to_csv(path_or_buf="tests/tmp_file.csv", columns=["text"])
    result = [
        "virginamerica sfo",
        "virginamerica",
        "virginamerica fly sfo seat",
        "fly virginamerica",
        "virginamerica fly",
        "virginamerica seat",
        "virginamerica love",
        "virginamerica love",
        "virginamerica",
        "virginamerica seat seat seat",
    ]

    assert (
        utils._prepare_corpus_path(
            text_corpus="tests/tmp_file.csv",
            target_cols="text",
            input_language="english",
            min_token_freq=2,
            min_token_len=3,
            min_tokens=0,
            max_token_index=-1,
            min_ngram_count=3,
            remove_stopwords=True,
            ignore_words=None,
            sample_size=1,
            verbose=False,
        )
        == result
    )

    os.remove("tests/tmp_file.csv")

    assert (
        utils._prepare_corpus_path(
            text_corpus=short_text_corpus,
            clean_texts=None,
            input_language="english",
            min_token_freq=2,
            min_token_len=3,
            min_tokens=0,
            max_token_index=-1,
            min_ngram_count=3,
            remove_stopwords=True,
            ignore_words=None,
            sample_size=1,
            verbose=False,
        )
        == result
    )
