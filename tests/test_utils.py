"""
Utilities Tests
---------------
"""

from kwx import utils


def test__combine_tokens_to_str():
    texts = ["words", "to", "not", "be", "combined"]
    texts_lol = [["words", "to"], ["not"], ["be", "combined"]]
    ignore_words = ["not"]
    result_ignore_not = "words to be combined"
    result_not = "words to not be combined"

    assert (
        utils._combine_tokens_to_str(texts=texts, ignore_words=ignore_words)
        == result_ignore_not
    )

    assert (
        utils._combine_tokens_to_str(texts=texts_lol, ignore_words=ignore_words)
        == result_ignore_not
    )

    assert utils._combine_tokens_to_str(texts=texts, ignore_words=None) == result_not

    assert (
        utils._combine_tokens_to_str(texts=texts_lol, ignore_words=None) == result_not
    )
