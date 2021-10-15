"""
utils
-----

Utility functions for data loading, cleaning, output formatting, and user interaction.

Contents:
    load_data,
    _combine_texts_to_str,
    _remove_unwanted,
    _lemmatize,
    clean,
    prepare_data,
    _prepare_corpus_path,
    translate_output,
    organize_by_pos,
    prompt_for_word_removal
"""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from collections import defaultdict

import gc
import os
import random
import string
from multiprocessing import Pool

import emoji
import gensim
import pandas as pd
import spacy
from googletrans import Translator
from nltk.stem.snowball import SnowballStemmer
from stopwordsiso import stopwords
from tqdm.auto import tqdm

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from gensim.models import Phrases

from kwx import languages


def load_data(data, target_cols=None):
    """
    Loads data from a path and formats it into a pandas df.

    Parameters
    ----------
        data : pd.DataFrame or csv/xlsx path
            The data in df or path form.

        target_cols : str or list (default=None)
            The columns in the csv/xlsx or dataframe that contain the text data to be modeled.

    Returns
    -------
        df_texts : pd.DataFrame
            The texts as a df.
    """
    if isinstance(data, str):
        if data[-len("xlsx") :] == "xlsx":
            df_texts = pd.read_excel(io=data)
        elif data[-len("csv") :] == "csv":
            df_texts = pd.read_csv(filepath_or_buffer=data)
        else:
            ValueError("Strings passed should be paths to csv or xlsx files.")

    elif isinstance(data, pd.DataFrame):
        df_texts = data

    elif isinstance(data, pd.Series):
        df_texts = pd.DataFrame(data).reset_index(drop=True)
        df_texts.columns = data.index.values.tolist()

    else:
        ValueError(
            "The 'data' argument should be either the name of a csv/xlsx file a pandas dataframe."
        )

    if target_cols is None:
        target_cols = df_texts.columns
    elif isinstance(target_cols, str):
        target_cols = [target_cols]

    df_texts = df_texts[target_cols]

    return df_texts


def _combine_texts_to_str(text_corpus, ignore_words=None):
    """
    Combines texts into one string.

    Parameters
    ----------
        text_corpus : str or list
            The texts to be combined.

        ignore_words : str or list
            Strings that should be removed from the text body.

    Returns
    -------
        texts_str : str
            A string of the full text with unwanted words removed.
    """
    if isinstance(ignore_words, str):
        words_to_ignore = [ignore_words]
    elif isinstance(ignore_words, list):
        words_to_ignore = ignore_words
    else:
        words_to_ignore = []

    if isinstance(text_corpus[0], list):
        flat_words = [text for sublist in text_corpus for text in sublist]
        flat_words = [
            token
            for subtext in flat_words
            for token in subtext.split(" ")
            if token not in words_to_ignore
        ]

    else:
        flat_words = [
            token
            for subtext in text_corpus
            for token in subtext.split(" ")
            if token not in words_to_ignore
        ]

    return " ".join(flat_words)


def _remove_unwanted(args):
    """
    Lower cases tokens and removes numbers and possibly names.

    Parameters
    ----------
        args : list of tuples
            The following arguments zipped.

        text : list
            The text to clean.

        words_to_ignore : str or list
            Strings that should be removed from the text body.

        stop_words : str or list
            Stopwords for the given language.

    Returns
    -------
        text_words_removed : list
            The text without unwanted tokens.
    """
    text, words_to_ignore, stop_words = args

    return [
        token.lower()
        for token in text
        if token not in words_to_ignore and token not in stop_words
    ]


def _lemmatize(tokens, nlp=None, verbose=True):
    """
    Lemmatizes tokens.

    Parameters
    ----------
        tokens : list or list of lists
            Tokens to be lemmatized.

        nlp : spacy.load object
            A spacy language model.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query.

    Returns
    -------
        base_tokens : list or list of lists
            Tokens that have been lemmatized for nlp analysis.
    """
    allowed_pos_tags = ["NOUN", "PROPN", "ADJ", "ADV", "VERB"]

    base_tokens = []
    for t in tqdm(
        tokens,
        total=len(tokens),
        desc="Texts lemmatized",
        unit="texts",
        disable=not verbose,
    ):
        combined_texts = _combine_texts_to_str(text_corpus=t)

        lem_tokens = nlp(combined_texts)
        lemmed_tokens = [
            token.lemma_ for token in lem_tokens if token.pos_ in allowed_pos_tags
        ]

        base_tokens.append(lemmed_tokens)

    return base_tokens


def clean(
    texts,
    input_language=None,
    min_token_freq=2,
    min_token_len=3,
    min_tokens=0,
    max_token_index=-1,
    min_ngram_count=3,
    remove_stopwords=True,
    ignore_words=None,
    sample_size=1,
    verbose=True,
):
    """
    Cleans and tokenizes a text body to prepare it for analysis.

    Parameters
    ----------
        texts : str or list
            The texts to be cleaned and tokenized.

        input_language : str (default=None)
            The English name of the language in which the texts are found.

        min_token_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus.

        min_token_len : int (default=3)
            The smallest allowable length of a word.

        min_tokens : int (default=0)
            The minimum allowable length of a tokenized text.

        max_token_index : int (default=-1)
            The maximum allowable length of a tokenized text.

        min_ngram_count : int (default=5)
            The minimum occurrences for an n-gram to be included.

        remove_stopwords : bool (default=True)
            Whether to remove stopwords.

        ignore_words : str or list
            Strings that should be removed from the text body.

        sample_size : float (default=1)
            The amount of data to be randomly sampled.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query.

    Returns
    -------
        text_corpus : list or list of lists
            The texts formatted for analysis.
    """
    input_language = input_language.lower()

    # Select abbreviation for the lemmatizer, if it's available.
    if input_language in languages.lem_abbr_dict().keys():
        input_language = languages.lem_abbr_dict()[input_language]

    if isinstance(texts, str):
        texts = [texts]

    if isinstance(ignore_words, str):
        words_to_ignore = [ignore_words]
    elif ignore_words is None:
        words_to_ignore = []
    else:
        words_to_ignore = ignore_words

    stop_words = []
    if remove_stopwords:
        if stopwords(input_language) != set():  # the input language has stopwords
            stop_words = stopwords(input_language)

        # Stemming and normal stopwords are still full language names.
        elif input_language in languages.stem_abbr_dict().keys():
            stop_words = stopwords(languages.stem_abbr_dict()[input_language])

        elif input_language in languages.sw_abbr_dict().keys():
            stop_words = stopwords(languages.sw_abbr_dict()[input_language])

    pbar = tqdm(
        desc="Cleaning steps complete", total=7, unit="step", disable=not verbose
    )
    # Remove spaces that are greater that one in length.
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
    # Prevent words from being combined when a user types word/word or word-word.
    for r in texts_no_large_spaces:
        r = r.replace("/", " ")
        r = r.replace("-", " ")
        if input_language == "fr":
            # Get rid of the 'of' abbreviation for French.
            r = r.replace("d'", "")

        texts_no_random_punctuation.append(r)

    texts_no_punctuation = [
        r.translate(str.maketrans("", "", string.punctuation + "–" + "’"))
        for r in texts_no_random_punctuation
    ]

    gc.collect()
    pbar.update()

    texts_no_emojis = [
        emoji.get_emoji_regexp().sub(r"", response) for response in texts_no_punctuation
    ]

    tokenized_texts = [
        [token for token in text.lower().split() if not token.isnumeric()]
        for text in texts_no_emojis
    ]
    tokenized_texts = [t for t in tokenized_texts if t != []]

    gc.collect()
    pbar.update()

    # Add bigrams and trigrams.
    # Use half the normal threshold.
    if float(gensim.__version__[0]) >= 4:
        bigrams = Phrases(
            sentences=tokenized_texts,
            min_count=min_ngram_count,
            threshold=5.0,
            connector_words=stop_words,
        )
        trigrams = Phrases(
            sentences=bigrams[tokenized_texts],
            min_count=min_ngram_count,
            threshold=5.0,
            connector_words=stop_words,
        )
    else:
        bigrams = Phrases(  # pylint: disable=unexpected-keyword-arg
            sentences=tokenized_texts,
            min_count=min_ngram_count,
            threshold=5.0,
            common_terms=stop_words,
        )
        trigrams = Phrases(  # pylint: disable=unexpected-keyword-arg
            sentences=bigrams[tokenized_texts],
            min_count=min_ngram_count,
            threshold=5.0,
            common_terms=stop_words,
        )

    tokens_with_ngrams = []
    for text in tqdm(
        tokenized_texts,
        total=len(tokenized_texts),
        desc="n-grams generated",
        unit="texts",
        disable=not verbose,
    ):
        for token in bigrams[text]:
            if token.count("_") == 1:
                # Token is a bigram, so add it to the tokens.
                text.insert(0, token)

        for token in trigrams[bigrams[text]]:
            if token.count("_") == 2:
                # Token is a trigram, so add it to the tokens.
                text.insert(0, token)

        tokens_with_ngrams.append(text)

    gc.collect()
    pbar.update()

    args = zip(
        tokens_with_ngrams,
        [words_to_ignore] * len(tokens_with_ngrams),
        [stop_words] * len(tokens_with_ngrams),
    )

    num_cores = os.cpu_count()
    if __name__ == "kwx.utils":
        with Pool(processes=num_cores) as pool:
            tokens_remove_unwanted = list(
                tqdm(
                    pool.imap(_remove_unwanted, args),
                    total=len(tokens_with_ngrams),
                    desc="Unwanted words removed",
                    unit="texts",
                    disable=not verbose,
                )
            )

    gc.collect()
    pbar.update()

    # Lemmatize or stem words (try the former first, then the latter).
    nlp = None
    try:
        nlp = spacy.load(input_language)
        base_tokens = _lemmatize(
            tokens=tokens_remove_unwanted, nlp=nlp, verbose=verbose
        )

    except OSError:
        try:
            os.system("python -m spacy download {}".format(input_language))
            nlp = spacy.load(input_language)
            base_tokens = _lemmatize(
                tokens=tokens_remove_unwanted, nlp=nlp, verbose=verbose
            )
        except OSError:
            nlp = None

    if nlp is None:
        # Lemmatization failed, so try stemming.
        stemmer = None
        if input_language in SnowballStemmer.languages:
            stemmer = SnowballStemmer(input_language)
        # Correct if the abbreviations were put in.
        elif input_language == "ar":
            stemmer = SnowballStemmer("arabic")
        elif input_language == "fi":
            stemmer = SnowballStemmer("finish")
        elif input_language == "hu":
            stemmer = SnowballStemmer("hungarian")
        elif input_language == "sv":
            stemmer = SnowballStemmer("swedish")

        if stemmer is None:
            # We cannot lemmatize or stem.
            base_tokens = tokens_remove_unwanted

        else:
            # Stemming instead of lemmatization.
            base_tokens = []  # still call it lemmatized for consistency.
            for tokens in tqdm(
                tokens_remove_unwanted,
                total=len(tokens_remove_unwanted),
                desc="Texts stemmed",
                unit="texts",
                disable=not verbose,
            ):
                stemmed_tokens = [stemmer.stem(t) for t in tokens]
                base_tokens.append(stemmed_tokens)

    gc.collect()
    pbar.update()

    # Remove words that don't appear enough or are too small.
    token_frequencies = defaultdict(int)
    for tokens in base_tokens:
        for t in list(set(tokens)):
            token_frequencies[t] += 1

    if min_token_len is None or min_token_len == False:
        min_token_len = 0
    if min_token_freq is None or min_token_freq == False:
        min_token_freq = 0

    assert isinstance(
        min_token_len, int
    ), "The 'min_token_len' argument must be an integer if used."
    assert isinstance(
        min_token_freq, int
    ), "The 'min_token_freq' argument must be an integer if used."

    min_len_freq_tokens = [
        [
            t
            for t in tokens
            if len(t) >= min_token_len and token_frequencies[t] >= min_token_freq
        ]
        for tokens in base_tokens
    ]

    gc.collect()
    pbar.update()

    # Derive those texts that still have valid words.
    non_empty_token_indexes = [i for i, t in enumerate(min_len_freq_tokens) if t != []]
    text_corpus = [min_len_freq_tokens[i] for i in non_empty_token_indexes]

    # Sample words, if necessary.
    if sample_size == 1:
        selected_idxs = list(range(len(text_corpus)))

    else:
        selected_idxs = [
            i
            for i in random.choices(
                range(len(text_corpus)), k=int(sample_size * len(text_corpus))
            )
        ]
    text_corpus = [
        _combine_texts_to_str(text_corpus=text_corpus[i]) for i in selected_idxs
    ]

    gc.collect()
    pbar.update()

    return text_corpus


def prepare_data(
    data=None,
    target_cols=None,
    input_language=None,
    min_token_freq=2,
    min_token_len=3,
    min_tokens=0,
    max_token_index=-1,
    min_ngram_count=3,
    remove_stopwords=True,
    ignore_words=None,
    sample_size=1,
    verbose=True,
):
    """
    Prepares input data for analysis from a pandas.DataFrame or path.

    Parameters
    ----------
        data : pd.DataFrame or csv/xlsx path
            The data in df or path form.

        target_cols : str or list (default=None)
            The columns in the csv/xlsx or dataframe that contain the text data to be modeled.

        input_language : str (default=None)
            The English name of the language in which the texts are found.

        min_token_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus.

        min_token_len : int (default=3)
            The smallest allowable length of a word.

        min_tokens : int (default=0)
            The minimum allowable length of a tokenized text.

        max_token_index : int (default=-1)
            The maximum allowable length of a tokenized text.

        min_ngram_count : int (default=5)
            The minimum occurrences for an n-gram to be included.

        remove_stopwords : bool (default=True)
            Whether to remove stopwords.

        ignore_words : str or list
            Strings that should be removed from the text body.

        sample_size : float (default=1)
            The amount of data to be randomly sampled.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query.

    Returns
    -------
        text_corpus, clean_texts, selected_idxs : list or list of lists, list, list
            The texts formatted for text analysis both as tokens and strings, as well as the indexes for selected entries.
    """
    input_language = input_language.lower()

    # Select abbreviation for the lemmatizer, if it's available.
    if input_language in languages.lem_abbr_dict().keys():
        input_language = languages.lem_abbr_dict()[input_language]

    if isinstance(target_cols, str):
        target_cols = [target_cols]

    df_texts = load_data(data)

    # Select columns from which texts should come.
    raw_texts = []
    for i in df_texts.index:
        text = "".join(
            " " + df_texts.loc[i, c]
            for c in target_cols
            if isinstance(df_texts.loc[i, c], str)
        )

        text = text[1:]  # remove first blank space
        raw_texts.append(text)

    return clean(
        texts=raw_texts,
        input_language=input_language,
        min_token_freq=min_token_freq,
        min_token_len=min_token_len,
        min_tokens=min_tokens,
        max_token_index=max_token_index,
        min_ngram_count=min_ngram_count,
        remove_stopwords=remove_stopwords,
        ignore_words=ignore_words,
        sample_size=sample_size,
        verbose=verbose,
    )


def _prepare_corpus_path(
    text_corpus=None,
    clean_texts=None,
    target_cols=None,
    input_language=None,
    min_token_freq=2,
    min_token_len=3,
    min_tokens=0,
    max_token_index=-1,
    min_ngram_count=3,
    remove_stopwords=True,
    ignore_words=None,
    sample_size=1,
    verbose=True,
):
    """
    Checks a text corpus to see if it's a path, and prepares the data if so.

    Parameters
    ----------
        text_corpus : str or list or list of lists
            A path or text corpus over which analysis should be done.

        clean_texts : str
            The texts formatted for analysis as strings.

        target_cols : str or list (default=None)
            The columns in the csv/xlsx or dataframe that contain the text data to be modeled.

        input_language : str (default=None)
            The English name of the language in which the texts are found.

        min_token_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus.

        min_token_len : int (default=3)
            The smallest allowable length of a word.

        min_tokens : int (default=0)
            The minimum allowable length of a tokenized text.

        max_token_index : int (default=-1)
            The maximum allowable length of a tokenized text.

        min_ngram_count : int (default=5)
            The minimum occurrences for an n-gram to be included.

        remove_stopwords : bool (default=True)
            Whether to remove stopwords.

        ignore_words : str or list
            Strings that should be removed from the text body.

        sample_size : float (default=1)
            The amount of data to be randomly sampled.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query.

    Returns
    -------
        text_corpus : list or list of lists
            A prepared text corpus for the data in the given path.
    """
    if isinstance(text_corpus, str):
        try:
            os.path.exists(text_corpus)  # a path has been provided
            text_corpus = prepare_data(
                data=text_corpus,
                target_cols=target_cols,
                input_language=input_language,
                min_token_freq=min_token_freq,
                min_token_len=min_token_len,
                min_tokens=min_tokens,
                max_token_index=max_token_index,
                min_ngram_count=min_ngram_count,
                remove_stopwords=remove_stopwords,
                ignore_words=ignore_words,
                sample_size=sample_size,
                verbose=verbose,
            )

            return text_corpus

        except OSError:
            return text_corpus

    return text_corpus


def translate_output(outputs, input_language, output_language):
    """
    Translates model outputs using https://github.com/ssut/py-googletrans.

    Parameters
    ----------
        outputs : list
            Output keywords of a model.

        input_language : str
            The English name of the language in which the texts are found.

        output_language
            The English name of the desired language for outputs.

    Returns
    -------
        translated_outputs : list
            A list of keywords translated to the given output_language.
    """
    translator = Translator()

    if isinstance(outputs[0], list):
        translated_outputs = [
            [
                translator.translate(
                    text=o, src=input_language, dest=output_language
                ).text
                for o in sub_output
            ]
            for sub_output in outputs
        ]

    elif isinstance(outputs[0], str):
        translated_outputs = [
            translator.translate(text=o, src=input_language, dest=output_language).text
            for o in outputs
        ]

    return translated_outputs


def organize_by_pos(outputs, output_language):
    """
    Orders a keyword output by the part of speech of the words.

    Parameters
    ----------
        outputs : list
            The keywords that have been extracted.

        output_language : str
            The spoken language in which the results should be given.

    Returns
    -------
        ordered_outputs : list
            The given keywords ordered by their pos.
    """
    if output_language in languages.lem_abbr_dict().keys():
        output_language = languages.lem_abbr_dict()[output_language]

    if (
        output_language in languages.lem_abbr_dict().values()
    ):  # we can use spacy to detect parts of speech.
        nlp = spacy.load(output_language)
        nlp_outputs = [nlp(o)[0] for o in outputs]

        # Those parts of speech to be considered (others go to an 'Other' category).
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


def prompt_for_word_removal(words_to_ignore=None):
    """
    Prompts the user for words that should be ignored in kewword extraction.

    Parameters
    ----------
        words_to_ignore : str or list
            Words that should not be included in the output.

    Returns
    -------
        ignore words, words_added : list, bool
            A new list of words to ignore and a boolean indicating if words have been added.
    """
    if isinstance(words_to_ignore, str):
        words_to_ignore = [words_to_ignore]

    words_to_ignore = [w.replace("'", "") for w in words_to_ignore]

    words_added = False  # whether to run the models again
    more_words = True
    while more_words:
        more_words = input("\nShould words be removed [y/n]? ")
        if more_words == "y":
            new_words_to_ignore = input("Type or copy word(s) to be removed: ")
            # Remove commas if the user has used them to separate words,
            # as well as apostraphes.
            new_words_to_ignore = [
                char for char in new_words_to_ignore if char not in [",", "'"]
            ]

            new_words_to_ignore = "".join(new_words_to_ignore)

            if " " in new_words_to_ignore:
                new_words_to_ignore = new_words_to_ignore.split(" ")
            elif isinstance(new_words_to_ignore, str):
                new_words_to_ignore = [new_words_to_ignore]

            words_to_ignore += new_words_to_ignore
            words_added = True  # we need to run the models again
            more_words = False

        elif more_words == "n":
            more_words = False

        else:
            print("Invalid input")

    return words_to_ignore, words_added
