# SPDX-License-Identifier: BSD-3-Clause
"""
Module for organizing language dependencies for text cleaning.

The following languages have been selected because their stopwords can be removed via https://github.com/stopwords-iso/stopwords-iso/tree/master/python.
"""


def lem_abbr_dict() -> dict[str, str]:
    """
    Call a dictionary of languages and their abbreviations for lemmatization.

    Returns
    -------
    dict
        A dictionary with languages as keys and their abbreviations as items.

    Notes
    -----
    These languages can be lemmatized via https://spacy.io/usage/models, and are also those that can have their words ordered by parts of speech.
    """
    return {
        "chinese": "zh",
        "danish": "da",
        "dutch": "nl",
        "english": "en",
        "french": "fr",
        "german": "de",
        "greek": "el",
        "italian": "it",
        "japanese": "ja",
        "lithuanian": "lt",
        "norwegian": "nb",
        "polish": "pl",
        "portuguese": "pt",
        "romanian": "ro",
        "spanish": "es",
    }


def stem_abbr_dict() -> dict[str, str]:
    """
    Call a dictionary of languages and their abbreviations for stemming.

    Returns
    -------
    dict
        A dictionary with languages as keys and their abbreviations as items.

    Notes
    -----
    These languages don't have good lemmatizers, and will thus be stemmed via https://www.nltk.org/api/nltk.stem.html.
    """
    return {
        "arabic": "ar",
        "finnish": "fi",
        "hungarian": "hu",
        "swedish": "sv",
    }


def sw_abbr_dict() -> dict[str, str]:
    """
    Call a dictionary of languages and their abbreviations for stop word removal.

    Returns
    -------
    dict
        A dictionary with languages as keys and their abbreviations as items.

    Notes
    -----
    These languages can only have their stopwords removed via https://github.com/stopwords-iso/stopwords-iso).
    """
    return {
        "afrikaans": "af",
        "bulgarian": "bg",
        "bengali": "bn",
        "breton": "br",
        "catalan": "ca",
        "czech": "cs",
        "esperanto": "eo",
        "estonian": "et",
        "basque": "eu",
        "farsi": "fa",
        "persian": "fa",
        "irish": "ga",
        "galician": "gl",
        "gujarati": "gu",
        "hausa": "ha",
        "hebrew": "he",
        "hindi": "hi",
        "croatian": "hr",
        "armenian": "hy",
        "indonesian": "id",
        "korean": "ko",
        "kurdish": "ku",
        "latin": "la",
        "latvian": "lv",
        "marathi": "mr",
        "malay": "ms",
        "norwegian": "no",
        "russian": "ru",
        "slovak": "sk",
        "slovenian": "sl",
        "somali": "so",
        "sotho": "st",
        "swahili": "sw",
        "thai": "th",
        "tagalog": "tl",
        "turkish": "tr",
        "ukrainian": "uk",
        "urdu": "ur",
        "vietnamese": "vi",
        "yoruba": "yo",
        "zulu": "zu",
    }
