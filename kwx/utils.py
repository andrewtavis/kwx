"""
utils
-----

Utility functions for data loading, cleaning, output formatting, and user interaction

Contents:
    load_data,
    _combine_tokens_to_str,
    _clean_text_strings,
    clean_and_tokenize_texts,
    prepare_data,
    _prepare_corpus_path,
    translate_output,
    organize_by_pos,
    prompt_for_word_removal
"""

import os
import re
import string
import random
from collections import defaultdict
import importlib

# Make sure xlrd is installed for pandas
xlrd_spec = importlib.util.find_spec("xlrd")
if xlrd_spec == None:
    os.system("pip install xlrd")

import pandas as pd

from nltk.stem.snowball import SnowballStemmer
import spacy

from googletrans import Translator
from stopwordsiso import stopwords
import emoji

from gensim.models import Phrases

from kwx import languages


def load_data(data, target_cols=None):
    """
    Loads data from a path and formats it into a pandas df

    Parameters
    ----------
        data : pd.DataFrame or csv/xlsx path
            The data in df or path form

        target_cols : str or list (default=None)
            The columns in the csv/xlsx or dataframe that contain the text data to be modeled

    Returns
    -------
        df_texts : pd.DataFrame
            The texts as a df
    """
    if type(data) == str:
        if data[-len("xlsx") :] == "xlsx":
            df_texts = pd.read_excel(io=data)
        elif data[-len("csv") :] == "csv":
            df_texts = pd.read_csv(filepath_or_buffer=data)
        else:
            ValueError("Strings passed should be csv or xlsx files.")

    elif type(data) == pd.DataFrame:
        df_texts = data

    else:
        ValueError(
            "The 'data' argument should be either the name of a csv/xlsx file a pandas dataframe."
        )

    if target_cols == None:
        target_cols = df_texts.columns
    elif type(target_cols) == str:
        target_cols = [target_cols]

    df_texts = df_texts[target_cols]

    return df_texts


def _combine_tokens_to_str(texts, ignore_words=None):
    """
    Combines the texts into one string

    Parameters
    ----------
        texts : str or list
            The texts to be combined

        ignore_words : str or list
            Strings that should be removed from the text body

    Returns
    -------
        texts_str : str
            A string of the full text with unwanted words removed
    """
    if type(texts[0]) == list:
        flat_words = [word for sublist in texts for word in sublist]
    else:
        flat_words = texts

    if type(ignore_words) == str:
        ignore_words = [ignore_words]
    elif ignore_words == None:
        ignore_words = []

    flat_words = [word for word in flat_words if word not in ignore_words]
    texts_str = " ".join([word for word in flat_words])

    return texts_str


def _clean_text_strings(s):
    """
    Cleans the string of a text body to prepare it for BERT analysis

    Parameters
    ----------
        s : str
            The combined texts to be cleaned

    Returns
    -------
        s : str
            The texts formatted for analysis
    """
    s = re.sub(r"([a-z])([A-Z])", r"\1\. \2", s)
    s = s.lower()
    s = re.sub(r"&gt|&lt", " ", s)
    s = re.sub(r"([a-z])\1{2,}", r"\1", s)
    s = re.sub(r"([\W+])\1{1,}", r"\1", s)
    s = re.sub(r"\*|\W\*|\*\W", ". ", s)
    s = re.sub(r"\(.*?\)", ". ", s)
    s = re.sub(r"\W+?\.", ".", s)
    s = re.sub(r"(\.|\?|!)(\w)", r"\1 \2", s)
    s = re.sub(r" ing ", " ", s)
    s = re.sub(r"product received for free[.| ]", " ", s)
    s = re.sub(r"(.{2,}?)\1{1,}", r"\1", s)

    return s.strip()


def lemmatize(tokens, nlp=None):
    """
    Lemmatizes tokens

    Parameters
    ----------
        tokens : list or list of lists

        nlp : spacy.load object
            A spacy language model

    Returns
    -------
        lemmatized_tokens : list or list of lists
            Tokens that have been lemmatized for nlp analysis
    """
    allowed_pos_tags = ["NOUN", "PROPN", "ADJ", "ADV", "VERB"]

    lemmatized_tokens = []
    for t in tokens:
        combined_tokens = _combine_tokens_to_str(texts=t)

        lem_tokens = nlp(combined_tokens)
        lemmed_tokens = [
            token.lemma_ for token in lem_tokens if token.pos_ in allowed_pos_tags
        ]

        lemmatized_tokens.append(lemmed_tokens)

    return lemmatized_tokens


def clean_and_tokenize_texts(
    texts, input_language=None, min_freq=2, min_word_len=3, sample_size=1
):
    """
    Cleans and tokenizes a text body to prepare it for analysis

    Parameters
    ----------
        texts : str or list
            The texts to be cleaned and tokenized

        input_language : str (default=None)
            The English name of the language in which the texts are found

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        sample_size : float (default=1)
            The amount of data to be randomly sampled

    Returns
    -------
        text_corpus, clean_texts, selection_idxs : list or list of lists, list, list
            The texts formatted for text analysis both as tokens and strings, as well as the indexes for selected entries
    """
    input_language = input_language.lower()

    # Select abbreviation for the lemmatizer, if it's available
    if input_language in languages.lem_abbr_dict().keys():
        input_language = languages.lem_abbr_dict()[input_language]

    if type(texts) == str:
        texts = [texts]

    # Remove spaces that are greater that one in length
    texts_no_large_spaces = []
    for r in texts:
        for i in range(
            25, 0, -1
        ):  # loop backwards to assure that smaller spaces aren't made
            large_space = str(i * " ")
            if large_space in r:
                r = r.replace(large_space, " ")

        texts_no_large_spaces.append(r)

    texts_no_random_punctuation = []
    # Prevent words from being combined when a user types word/word or word-word
    for r in texts_no_large_spaces:
        r = r.replace("/", " ")
        r = r.replace("-", " ")
        if input_language == "fr":
            # Get rid of the 'of' abbreviation for French
            r = r.replace("d'", "")

        texts_no_random_punctuation.append(r)

    # Remove punctuation
    texts_no_punctuation = []
    for r in texts_no_random_punctuation:
        texts_no_punctuation.append(
            r.translate(str.maketrans("", "", string.punctuation + "–" + "’"))
        )

    # Remove emojis
    texts_no_emojis = []
    for response in texts_no_punctuation:
        texts_no_emojis.append(emoji.get_emoji_regexp().sub(r"", response))

    # Remove stopwords and tokenize
    if stopwords(input_language) != set():  # the input language has stopwords
        stop_words = stopwords(input_language)
    # Stemming and normal stopwords are still full language names
    elif input_language in languages.stem_abbr_dict().keys():
        stop_words = stopwords(languages.stem_abbr_dict()[input_language])
    elif input_language in languages.sw_abbr_dict().keys():
        stop_words = stopwords(languages.sw_abbr_dict()[input_language])
    else:
        stop_words = []

    tokenized_texts = [
        [
            word
            for word in text.lower().split()
            if word not in stop_words and not word.isnumeric()
        ]
        for text in texts_no_emojis
    ]
    tokenized_texts = [t for t in tokenized_texts if t != []]

    # Add bigrams (first_second word combinations that appear often together)
    tokens_with_bigrams = []
    bigrams = Phrases(
        sentences=tokenized_texts, min_count=3, threshold=5.0
    )  # minimum count for a bigram to be included is 3
    for i, t in enumerate(tokenized_texts):
        for token in bigrams[t]:
            if "_" in token:
                # Token is a bigram, so add it to the tokens
                t.insert(0, token)

        tokens_with_bigrams.append(t)

    # Lemmatize or stem words (try the former first, then the latter)
    nlp = None
    try:
        nlp = spacy.load(input_language)
        lemmatized_tokens = lemmatize(tokens=tokens_with_bigrams, nlp=nlp)

    except OSError:
        try:
            os.system("python -m spacy download {}".format(input_language))
            nlp = spacy.load(input_language)
            lemmatized_tokens = lemmatize(tokens=tokens_with_bigrams, nlp=nlp)
        except:
            pass

    if nlp == None:
        # Lemmatization failed, so try stemmiing
        stemmer = None
        if input_language in SnowballStemmer.languages:
            stemmer = SnowballStemmer(input_language)
        # Correct if the abbreviations were put in
        elif input_language == "ar":
            stemmer = SnowballStemmer("arabic")
        elif input_language == "fi":
            stemmer = SnowballStemmer("finish")
        elif input_language == "hu":
            stemmer = SnowballStemmer("hungarian")
        elif input_language == "sv":
            stemmer = SnowballStemmer("swedish")

        if stemmer != None:
            # Stemming instead of lemmatization
            lemmatized_tokens = []  # still call it lemmatized for consistency
            for tokens in tokens_with_bigrams:
                stemmed_tokens = [stemmer.stem(t) for t in tokens]
                lemmatized_tokens.append(stemmed_tokens)

        else:
            # We cannot lemmatize or stem
            lemmatized_tokens = tokens_with_bigrams

    # Remove words that don't appear enough or are too small
    token_frequencies = defaultdict(int)
    for tokens in lemmatized_tokens:
        for t in list(set(tokens)):
            token_frequencies[t] += 1

    if min_word_len == None or min_word_len == False:
        min_word_len = 0
    if min_freq == None or min_freq == False:
        min_freq = 0

    min_len_freq_tokens = []
    for tokens in lemmatized_tokens:
        min_len_freq_tokens.append(
            [
                t
                for t in tokens
                if len(t) >= min_word_len and token_frequencies[t] >= min_freq
            ]
        )

    # Derive those texts that still have valid words
    non_empty_token_indexes = [i for i, t in enumerate(min_len_freq_tokens) if t != []]
    text_corpus = [min_len_freq_tokens[i] for i in non_empty_token_indexes]
    clean_texts = [_clean_text_strings(s=texts[i]) for i in non_empty_token_indexes]

    # Sample words, if necessary
    if sample_size != 1:
        selected_idxs = [
            i
            for i in random.choices(
                range(len(text_corpus)), k=int(sample_size * len(text_corpus))
            )
        ]
    else:
        selected_idxs = list(range(len(text_corpus)))

    text_corpus = [text_corpus[i] for i in selected_idxs]
    clean_texts = [clean_texts[i] for i in selected_idxs]

    return text_corpus, clean_texts, selected_idxs


def prepare_data(
    data=None,
    target_cols=None,
    input_language=None,
    min_freq=2,
    min_word_len=3,
    sample_size=1,
):
    """
    Prepares input data for analysis

    Parameters
    ----------
        data : pd.DataFrame or csv/xlsx path
            The data in df or path form

        target_cols : str or list (default=None)
            The columns in the csv/xlsx or dataframe that contain the text data to be modeled

        input_language : str (default=None)
            The English name of the language in which the texts are found

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        sample_size : float (default=1)
            The amount of data to be randomly sampled

    Returns
    -------
        text_corpus, clean_texts, selected_idxs : list or list of lists, list, list
            The texts formatted for text analysis both as tokens and strings, as well as the indexes for selected entries
    """
    input_language = input_language.lower()

    # Select abbreviation for the lemmatizer, if it's available
    if input_language in languages.lem_abbr_dict().keys():
        input_language = languages.lem_abbr_dict()[input_language]

    if type(target_cols) == str:
        target_cols = [target_cols]

    df_texts = load_data(data)

    # Select columns from which texts should come
    raw_texts = []

    for i in df_texts.index:
        text = ""
        for c in target_cols:
            if type(df_texts.loc[i, c]) == str:
                text += " " + df_texts.loc[i, c]

        text = text[1:]  # remove first blank space
        raw_texts.append(text)

    text_corpus, clean_texts, selected_idxs = clean_and_tokenize_texts(
        texts=raw_texts,
        input_language=input_language,
        min_freq=min_freq,
        min_word_len=min_word_len,
        sample_size=sample_size,
    )

    return text_corpus, clean_texts, selected_idxs


def _prepare_corpus_path(
    text_corpus=None,
    clean_texts=None,
    target_cols=None,
    input_language=None,
    min_freq=2,
    min_word_len=3,
    sample_size=1,
):
    """
    Checks a text corpus to see if it's a path, and prepares the data if so

    Parameters
    ----------
        text_corpus : str or list or list of lists
            A path or text corpus over which analysis should be done

        clean_texts : str
            The texts formatted for analysis as strings

        target_cols : str or list (default=None)
            The columns in the csv/xlsx or dataframe that contain the text data to be modeled

        input_language : str (default=None)
            The English name of the language in which the texts are found

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        sample_size : float (default=1)
            The amount of data to be randomly sampled

    Returns
    -------
        text_corpus : list or list of lists
            A prepared text corpus for the data in the given path
    """
    if type(text_corpus) == str:
        try:
            os.path.exists(text_corpus)  # a path has been provided
            text_corpus, clean_texts = prepare_data(
                data=text_corpus,
                target_cols=target_cols,
                input_language=input_language,
                min_freq=min_freq,
                min_word_len=min_word_len,
                sample_size=sample_size,
            )[:2]

            return text_corpus, clean_texts

        except:
            pass

    if clean_texts != None:
        return text_corpus, clean_texts

    else:
        return (
            text_corpus,
            [
                _clean_text_strings(_combine_tokens_to_str(texts=t_c))
                for t_c in text_corpus
            ],
        )


def translate_output(outputs, input_language, output_language):
    """
    Translates model outputs using https://github.com/ssut/py-googletrans

    Parameters
    ----------
        outputs : list
            Output keywords of a model

        input_language : str
            The English name of the language in which the texts are found

        output_language
            The English name of the desired language for outputs

    Returns
    -------
        translated_outputs : list
            A list of keywords translated to the given output_language
    """
    translator = Translator()

    if type(outputs[0]) == list:
        translated_outputs = []
        for sub_output in outputs:
            translated_outputs.append(
                [
                    translator.translate(
                        text=o, src=input_language, dest=output_language
                    ).text
                    for o in sub_output
                ]
            )

    elif type(outputs[0]) == str:
        translated_outputs = [
            translator.translate(text=o, src=input_language, dest=output_language).text
            for o in outputs
        ]

    return translated_outputs


def organize_by_pos(outputs, output_language):
    """
    Orders a keyword output by the part of speech of the words

    Order is: nouns, adjectives, adverbs and verbs

    Parameters
    ----------
        outputs : list
            The keywords that have been extracted

        output_language : str
            The spoken language in which the results should be given

    Returns
    -------
        ordered_outputs : list
            The given keywords ordered by their pos
    """
    if output_language in languages.lem_abbr_dict().keys():
        output_language = languages.lem_abbr_dict()[output_language]

    if (
        output_language in languages.lem_abbr_dict().values()
    ):  # we can use spacy to detect parts of speech
        nlp = spacy.load(output_language)
        nlp_outputs = [nlp(o)[0] for o in outputs]

        # Those parts of speech to be considered (others go to an 'Other' category)
        pos_order = ["NOUN", "PROPN", "ADJ", "ADV", "VERB"]
        ordered_outputs = [[o for o in nlp_outputs if o.pos_ == p] for p in pos_order]
        flat_ordered_outputs = [str(o) for sub in ordered_outputs for o in sub]

        other = []
        for o in outputs:
            if o not in flat_ordered_outputs:
                other.append(o)
        ordered_outputs.append(other)

        outputs_dict = {}
        for i, o in enumerate(ordered_outputs):
            if i == 0:
                outputs_dict["Nouns:"] = o
            if i == 1:
                outputs_dict["Nouns:"] += o  # proper nouns put in nouns
            if i == 2:
                outputs_dict["Adjectives:"] = ordered_outputs[i]
            if i == 3:
                outputs_dict["Adverbs:"] = ordered_outputs[i]
            if i == 4:
                outputs_dict["Verbs:"] = ordered_outputs[i]
            if i == 5:
                outputs_dict["Other:"] = ordered_outputs[i]

        outputs_dict = {
            k: v for k, v in outputs_dict.items() if v != []
        }  # remove if no entries

        return outputs_dict

    else:
        return outputs


def prompt_for_word_removal(ignore_words=None):
    """
    Prompts the user for words that should be ignored in kewword extraction

    Parameters
    ----------
        ignore_words : str or list
            Words that should not be included in the output

    Returns
    -------
        ignore words, words_added : list, bool
            A new list of words to ignore and a boolean indicating if words have been added
    """
    if ignore_words == None:
        ignore_words = []
    if type(ignore_words) == str:
        ignore_words = [ignore_words]

    ignore_words = [w.replace("'", "") for w in ignore_words]

    words_added = False  # whether to run the models again
    more_words = True
    while more_words != False:
        more_words = input("\nAre there words that should be removed [y/n]? ")
        if more_words == "y":
            new_words_to_ignore = input("Type or copy word(s) to be removed: ")
            # Remove commas if the user has used them to separate words, as well as apostraphes
            new_words_to_ignore = [
                char for char in new_words_to_ignore if char != "," and char != "'"
            ]
            new_words_to_ignore = "".join([word for word in new_words_to_ignore])

            if " " in new_words_to_ignore:
                new_words_to_ignore = new_words_to_ignore.split(" ")
            elif type(new_words_to_ignore) == str:
                new_words_to_ignore = [new_words_to_ignore]

            ignore_words += new_words_to_ignore
            words_added = True  # we need to run the models again
            more_words = False

        elif more_words == "n":
            more_words = False

        else:
            print("Invalid input")

    return ignore_words, words_added
