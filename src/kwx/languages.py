"""
languages
---------

Module for organizing language dependencies for text cleaning.

The following languages have been selected because their stopwords can be removed via https://github.com/stopwords-iso/stopwords-iso/tree/master/python.

Contents:
    lem_abbr_dict,
    stem_abbr_dict,
    sw_abbr_dict
"""


def lem_abbr_dict():
    """
    Calls a dictionary of languages and their abbreviations for lemmatization.

    Notes
    -----
        These languages can be lemmatized via https://spacy.io/usage/models, and are also those that can have their words ordered by parts of speech.

    Returns
    -------
        lem_abbr_dict : dict
            A dictionary with languages as keys and their abbreviations as items.
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


def stem_abbr_dict():
    """
    Calls a dictionary of languages and their abbreviations for stemming.

    Notes
    -----
        These languages don't have good lemmatizers, and will thus be stemmed via https://www.nltk.org/api/nltk.stem.html.

    Returns
    -------
        stem_abbr_dict : dict
            A dictionary with languages as keys and their abbreviations as items.
    """
    return {
        "arabic": "ar",
        "finnish": "fi",
        "hungarian": "hu",
        "swedish": "sv",
    }


def sw_abbr_dict():
    """
    Calls a dictionary of languages and their abbreviations for stop word removal.

    Notes
    -----
        These languages can only have their stopwords removed via https://github.com/stopwords-iso/stopwords-iso).

    Returns
    -------
        sw_abbr_dict : dict
            A dictionary with languages as keys and their abbreviations as items.
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
