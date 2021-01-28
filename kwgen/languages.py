"""
Languages
---------

Module for organizing language dependencies for text cleaning

Contents
    lem_lang_abbr_dict,
    stem_lang_abbr_dict,
    stop_wrods_lang_abbr_dict
"""


# The following languages have been selected because:
# Their stopwords can be removed via https://github.com/stopwords-iso/stopwords-iso/tree/master/python

# Their following can be lemmatized via https://spacy.io/usage/models
# These languages are also those that can have their words ordered by part of speech


def lem_lang_abbr_dict():
    lem_lang_abbr_dict = {
        "chinese": "zh",
        "danish": "da",
        "dutch": "nl",
        "english": "en",
        "french": "fr",
        "german": "de",
        "greek": "el",
        "italian": "it",
        "japaneze": "ja",
        "lithuanian": "lt",
        "norwegian": "nb",
        "polish": "pl",
        "portugese": "pt",
        "romanian": "ro",
        "spanish": "es",
    }

    return lem_lang_abbr_dict


# Hungarian and other languages don't have good lemmatizers, and will thus be stemmed via: https://www.nltk.org/api/nltk.stem.html
def stem_lang_abbr_dict():
    stem_lang_abbr_dict = {
        "arabic": "ar",
        "finnish": "fi",
        "hungarian": "hu",
        "swedish": "sv",
    }

    return stem_lang_abbr_dict


# The final languages can only have their stopwords removed (see https://github.com/stopwords-iso/stopwords-iso)
def stop_wrods_lang_abbr_dict():
    stop_wrods_lang_abbr_dict = {
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

    return stop_wrods_lang_abbr_dict
