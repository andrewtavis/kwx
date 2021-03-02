"""
model
-----

Functions for modeling text corpuses and extracting keywords

Contents:
    get_topic_words,
    get_coherence,
    _order_and_subset_by_coherence,
    extract_kws,
    gen_files
"""

import os
import math
import re
import inspect
import zipfile
from collections import Counter
import importlib

import numpy as np
import pandas as pd

from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from sentence_transformers import SentenceTransformer

from kwx import utils, languages, topic_model, visuals


def get_topic_words(text_corpus, labels, num_topics=None, num_keywords=None):
    """
    Get top words within each topic for cluster models

    Parameters
    ----------
        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done

        labels : list
            The labels assigned to topics

        num_topics : int (default=None)
            The number of categories for LDA and BERT based approaches

        num_keywords : int (default=None)
            The number of keywords that should be extracted

    Returns
    -------
        topics, non_blank_topic_idxs : list and list
            Topic keywords and indexes of those that are not empty lists
    """
    if num_topics == None:
        num_topics = len(np.unique(labels))
    topics = ["" for _ in range(num_topics)]
    for i, c in enumerate(text_corpus):
        topics[labels[i]] += " " + " ".join(c)

    # Count the words that appear for a given topic label
    word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
    word_counts = list(
        map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts)
    )

    topics = list(
        map(lambda x: list(map(lambda x: x[0], x[:num_keywords])), word_counts)
    )

    non_blank_topic_idxs = [i for i, t in enumerate(topics) if t != []]
    topics = [topics[i] for i in non_blank_topic_idxs]

    return topics, non_blank_topic_idxs


def get_coherence(model, text_corpus, num_topics=10, num_keywords=10, measure="c_v"):
    """
    Gets model coherence from gensim.models.coherencemodel

    Parameters
    ----------
        model : kwx.topic_model.TopicModel
            A model trained on the given text corpus

        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done

        num_topics : int (default=10)
            The number of categories for LDA and BERT based approaches

        num_keywords : int (default=10)
            The number of keywords that should be extracted

        measure : str (default=c_v)
            A gensim measure of coherence

    Returns
    -------
        coherence : float
            The coherence of the given model over the given texts
    """
    if model.method.lower() == "lda":
        cm = CoherenceModel(
            model=model.lda_model,
            texts=text_corpus,
            corpus=model.bow_corpus,
            dictionary=model.dirichlet_dict,
            coherence=measure,
        )
    else:
        topic_words = get_topic_words(
            text_corpus=text_corpus,
            labels=model.cluster_model.labels_,
            num_topics=num_topics,
            num_keywords=num_keywords,
        )[0]

        cm = CoherenceModel(
            topics=topic_words,
            texts=text_corpus,
            corpus=model.bow_corpus,
            dictionary=model.dirichlet_dict,
            coherence=measure,
        )

    coherence = cm.get_coherence()

    return coherence


def _order_and_subset_by_coherence(model, num_topics=10, num_keywords=10):
    """
    Orders topics based on their average coherence across the text corpus

    Parameters
    ----------
        model : kwx.topic_model.TopicModel
            A model trained on the given text corpus

        num_topics : int (default=10)
            The number of categories for LDA and BERT based approaches

        num_keywords : int (default=10)
            The number of keywords that should be extracted

    Returns
    -------
        ordered_topic_words, selection_indexes: list of lists and list of lists
            Topics words ordered by average coherence and indexes by which they should be selected
    """
    # Derive average topics across texts for a given method
    if model.method == "lda":
        shown_topics = model.lda_model.show_topics(
            num_topics=num_topics, num_words=num_keywords, formatted=False
        )

        topic_words = [[word[0] for word in topic[1]] for topic in shown_topics]
        topic_corpus = model.lda_model.__getitem__(
            bow=model.bow_corpus, eps=0
        )  # cutoff probability to 0

        topics_per_response = [response for response in topic_corpus]
        flat_topic_coherences = [
            item for sublist in topics_per_response for item in sublist
        ]

        topic_averages = [
            (
                t,
                sum([t_c[1] for t_c in flat_topic_coherences if t_c[0] == t])
                / len(model.bow_corpus),
            )
            for t in range(num_topics)
        ]

    elif model.method == "bert" or model.method == "lda_bert":
        # The topics in cluster models are not guranteed to be the size of num_keywords
        topic_words, non_blank_topic_idxs = get_topic_words(
            text_corpus=model.text_corpus,
            labels=model.cluster_model.labels_,
            num_topics=num_topics,
            num_keywords=num_keywords,
        )

        # Create a dictionary of the assignment counts for the topics
        counts_dict = dict(Counter(model.cluster_model.labels_))
        counts_dict = {
            k: v for k, v in counts_dict.items() if k in non_blank_topic_idxs
        }
        keys_ordered = sorted([k for k in counts_dict.keys()])

        # Map to the range from 0 to the number of non-blank topics
        counts_dict_mapped = {i: counts_dict[k] for i, k in enumerate(keys_ordered)}

        # Derive the average assignment of the topics
        topic_averages = [
            (k, counts_dict_mapped[k] / sum(counts_dict_mapped.values()))
            for k in counts_dict_mapped.keys()
        ]

    # Order ids by the average coherence across the texts
    topic_ids_ordered = [
        tup[0] for tup in sorted(enumerate(topic_averages), key=lambda i: i[1][1])[::-1]
    ]
    ordered_topic_words = [topic_words[i] for i in topic_ids_ordered]

    ordered_topic_averages = [
        tup[1] for tup in sorted(topic_averages, key=lambda i: i[1])[::-1]
    ]
    ordered_topic_averages = [
        a / sum(ordered_topic_averages) for a in ordered_topic_averages
    ]  # normalize just in case

    # Create selection indexes for each topic given its average coherence and how many keywords are wanted
    selection_indexes = [
        list(range(int(math.floor(num_keywords * a))))
        if math.floor(num_keywords * a) > 0
        else [0]
        for i, a in enumerate(ordered_topic_averages)
    ]

    total_indexes = sum([len(i) for i in selection_indexes])
    s_i = 0
    while total_indexes < num_keywords:
        selection_indexes[s_i] = selection_indexes[s_i] + [
            selection_indexes[s_i][-1] + 1
        ]
        s_i += 1
        total_indexes += 1

    return ordered_topic_words, selection_indexes


def extract_kws(
    method="lda",
    text_corpus=None,
    clean_texts=None,
    input_language=None,
    output_language=None,
    num_keywords=10,
    num_topics=10,
    corpuses_to_compare=None,
    return_topics=False,
    ignore_words=None,
    min_freq=2,
    min_word_len=3,
    sample_size=1,
    prompt_remove_words=True,
):
    """
    Extracts keywords given data, metadata, and model parameter inputs

    Parameters
    ----------
        method : str (default=lda)
            The modelling method

            Options:
                frequency: a count of the most frequent words

                TFIDF: Term Frequency Inverse Document Frequency

                    - Allows for words within one text group to be compared to those of another
                    - Gives a better idea of what users specifically want from a given publication

                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories
                    - These categories are then used to classify individual entries given the percent they fall into categories

                BERT: Bidirectional Encoder Representations from Transformers

                    - Words are classified via Google Neural Networks
                    - Word classifications are then used to derive topics

                LDA_BERT: Latent Dirichlet Allocation with BERT embeddigs

                    - The combination of LDA and BERT via an autoencoder

        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done

            Note 1: generated using prepare_text_data

            Note 2: if a str is provided, then the data will be loaded from a path

        clean_texts : list
            Text strings that are formatted for cluster models

        input_language : str (default=None)
            The spoken language in which the texts are found

        output_language : str (default=None: same as input_language)
            The spoken language in which the results should be given

        num_topics : int (default=10)
            The number of categories for LDA and BERT based approaches

        num_keywords : int (default=10)
            The number of keywords that should be extracted

        corpuses_to_compare : list : contains lists (default=None)
            A list of other text corpuses that the main corpus should be compared to using TFIDF

        return_topics : bool (default=False)
            Whether to return the topics that are extracted by an LDA model

        ignore_words : str or list (default=None)
            Words that should be removed (such as the name of the publisher)

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        sample_size : float (default=1)
            The amount of data to be randomly sampled

        prompt_remove_words : bool (default=True)
            Whether to prompt the user for keywords to remove

    Returns
    -------
        output_keywords : list or list of lists

            - A list of togs that should be used to present the publisher

            - A list of lists where sub_lists are the keywords best associated with the data entry
    """
    input_language = input_language.lower()
    method = method.lower()

    valid_methods = ["frequency", "tfidf", "lda", "bert", "lda_bert"]

    assert (
        method in valid_methods
    ), "The value for the 'method' argument is invalid. Please choose one of {}.".format(
        valid_methods
    )

    if method.lower() == "tfidf":
        assert (
            corpuses_to_compare != None
        ), "TFIDF requires another text corpus to be passed to the `corpuses_to_compare` argument."

    if input_language in languages.lem_abbr_dict().keys():
        input_language = languages.lem_abbr_dict()[input_language]

    if output_language == None:
        output_language = input_language
    else:
        output_language = output_language.lower()
        if output_language in languages.lem_abbr_dict().keys():
            output_language = languages.lem_abbr_dict()[output_language]

    if ignore_words is not None:
        if type(ignore_words) == str:
            ignore_words = [ignore_words]
    else:
        ignore_words = []

    # Generate text corpus from df or a path
    text_corpus, clean_texts = utils._prepare_corpus_path(
        text_corpus=text_corpus,
        clean_texts=clean_texts,
        input_language=input_language,
        min_freq=min_freq,
        min_word_len=min_word_len,
        sample_size=sample_size,
    )

    if method == "frequency" or method == "tfidf":
        if method == "frequency":
            word_counts = Counter(item for sublist in text_corpus for item in sublist)
            sorted_word_counts = {
                k: v
                for k, v in sorted(word_counts.items(), key=lambda item: item[1])[::-1]
            }
            top_word_counts = {
                k: v
                for k, v in sorted_word_counts.items()
                if k
                in [
                    key for key in sorted_word_counts.keys() if key not in ignore_words
                ][:num_keywords]
            }

            keywords = list(top_word_counts.keys())

        elif method == "tfidf":  # Term Frequency Inverse Document Frequency
            # Format the other texts to compare
            if type(corpuses_to_compare) == str:
                try:
                    os.path.exists(corpuses_to_compare)  # a path has been provided
                    corpuses_to_compare = utils.prepare_data(
                        data=corpuses_to_compare,
                        input_language=input_language,
                        min_freq=min_freq,
                        min_word_len=min_word_len,
                        sample_size=sample_size,
                    )[0]
                except:
                    pass

            elif type(corpuses_to_compare) == list:
                try:
                    os.path.exists(corpuses_to_compare[0])
                    corpus_paths = [c for c in corpuses_to_compare]
                    for c in corpus_paths:
                        corpuses_to_compare.append(
                            utils.prepare_data(
                                data=c,
                                input_language=input_language,
                                min_freq=min_freq,
                                min_word_len=min_word_len,
                                sample_size=sample_size,
                            )[0]
                        )

                    corpuses_to_compare = [
                        c for c in corpuses_to_compare if c not in corpus_paths
                    ][0]

                except:
                    pass

            if type(corpuses_to_compare[0]) == str:  # only one corpus to compare
                corpuses_to_compare = [corpuses_to_compare]

            # Combine the main corpus and those to compare
            comparative_corpus = [corpuses_to_compare]
            comparative_corpus.insert(0, text_corpus)

            comparative_string_corpus = []
            for c in comparative_corpus:
                combined_tokens = utils._combine_tokens_to_str(c)

                comparative_string_corpus.append(combined_tokens)

            tfidf_vectorizer = TfidfVectorizer(
                use_idf=True, smooth_idf=True, sublinear_tf=True
            )
            model = tfidf_vectorizer.fit_transform(  # pylint: disable=unused-variable
                comparative_string_corpus
            )
            corpus_scored = tfidf_vectorizer.transform(comparative_string_corpus)
            terms = tfidf_vectorizer.get_feature_names()
            scores = corpus_scored.toarray().flatten().tolist()
            keywords_and_scores = list(zip(terms, scores))

            keywords = [
                word[0]
                for word in sorted(
                    keywords_and_scores, key=lambda x: x[1], reverse=True
                )[:num_keywords]
                if word not in ignore_words
            ]

            # Check that more words than the number that appear in the text is not given
            frequent_words = extract_kws(
                method="frequency",
                text_corpus=text_corpus,
                input_language=input_language,
                output_language=output_language,
                num_keywords=num_keywords,
                num_topics=num_topics,
                corpuses_to_compare=corpuses_to_compare,
                return_topics=False,
                ignore_words=ignore_words,
                min_freq=min_freq,
                min_word_len=min_word_len,
                sample_size=sample_size,
                prompt_remove_words=False,  # prevent recursion
            )

            if len(keywords) > len(frequent_words):
                keywords = keywords[: len(frequent_words)]

    elif method == "lda" or method == "bert" or method == "lda_bert":
        # Create and fit a topic model on the data
        bert_model = None
        if method == "bert" or method == "lda_bert":
            # Multilingual BERT model trained on the top 100+ Wikipedias for semantic textual similarity
            bert_model = SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")

        tm = topic_model.TopicModel(
            num_topics=num_topics, method=method, bert_model=bert_model
        )
        tm.fit(
            texts=clean_texts, text_corpus=text_corpus, method=method, m_clustering=None
        )

        ordered_topic_words, selection_indexes = _order_and_subset_by_coherence(
            model=tm, num_topics=num_topics, num_keywords=num_keywords
        )

        if return_topics:
            # Return topics to inspect them
            if output_language != input_language:
                ordered_topic_words = utils.translate_output(
                    outputs=ordered_topic_words,
                    input_language=input_language,
                    output_language=output_language,
                )

            return ordered_topic_words

        else:
            # Reverse all selection variables so that low level words come from strong topics
            selection_indexes = selection_indexes[::-1]
            ordered_topic_words = ordered_topic_words[::-1]

            flat_ordered_topic_words = [
                word for topic in ordered_topic_words for word in topic
            ]
            set_ordered_topic_words = list(set(flat_ordered_topic_words))
            set_ordered_topic_words = [
                t_w for t_w in set_ordered_topic_words if t_w not in ignore_words
            ]
            if len(set_ordered_topic_words) <= num_keywords:
                print("\n")
                print(
                    "WARNING: the number of distinct topic words is less than the desired number of keywords."
                )
                print("All topic words will be returned.")
                keywords = set_ordered_topic_words

            else:
                # Derive keywords from Dirichlet or cluster algorithms
                t_n = 0
                keywords = []
                while len(keywords) < num_keywords:
                    sel_idxs = selection_indexes[t_n]

                    for s_i in sel_idxs:
                        if (
                            ordered_topic_words[t_n][s_i] not in keywords
                            and ordered_topic_words[t_n][s_i] not in ignore_words
                        ):
                            keywords.append(ordered_topic_words[t_n][s_i])
                        else:
                            sel_idxs.append(sel_idxs[-1] + 1)

                        if len(sel_idxs) >= len(ordered_topic_words[t_n]):
                            # The indexes are now more than the keywords, so move to the next topic
                            break

                    t_n += 1
                    if t_n == len(ordered_topic_words):
                        # The last topic has been gone through, so return to the first
                        t_n = 0

            # Fix for if too many were selected
            keywords = keywords[:num_keywords]

            # As a final check, if there are not enough words, then add non-included most frequent ones in order
            if len(keywords) < num_keywords:
                frequent_words = extract_kws(
                    method="frequency",
                    text_corpus=text_corpus,
                    input_language=input_language,
                    output_language=output_language,
                    num_keywords=num_keywords,
                    num_topics=num_topics,
                    corpuses_to_compare=corpuses_to_compare,
                    return_topics=False,
                    ignore_words=ignore_words,
                    min_freq=min_freq,
                    min_word_len=min_word_len,
                    sample_size=sample_size,
                    prompt_remove_words=False,  # prevent recursion
                )

                for word in frequent_words:
                    if word not in keywords and len(keywords) < len(frequent_words):
                        keywords.append(word)

    if prompt_remove_words:
        # Ask user if words should be ignored, and iterate until no more words should be
        more_words_to_ignore = True
        first_iteration = True
        new_words_to_ignore = ignore_words  # initialize so that it can be added to
        while more_words_to_ignore != False:
            if first_iteration == True:
                print("The {} keywords are:\n".format(method.upper()))
                print(keywords)
            else:
                print("\n")
                print("The new {} keywords are:\n".format(method.upper()))
                print(keywords)

            new_words_to_ignore, words_added = utils.prompt_for_word_removal(
                ignore_words=new_words_to_ignore
            )
            first_iteration = False

            if words_added == True:
                keywords = extract_kws(
                    method=method,
                    text_corpus=text_corpus,
                    clean_texts=clean_texts,
                    input_language=input_language,
                    output_language=output_language,
                    num_keywords=num_keywords,
                    num_topics=num_topics,
                    corpuses_to_compare=corpuses_to_compare,
                    return_topics=False,
                    ignore_words=new_words_to_ignore,
                    min_freq=min_freq,
                    min_word_len=min_word_len,
                    sample_size=sample_size,
                    prompt_remove_words=False,  # prevent recursion
                )

            else:
                more_words_to_ignore = False

    if output_language != input_language:
        translated_keywords = utils.translate_output(
            outputs=keywords,
            input_language=input_language,
            output_language=output_language,
        )

        return translated_keywords

    else:
        return keywords


def gen_files(
    method=["lda", "lda_bert"],
    text_corpus=None,
    clean_texts=None,
    input_language=None,
    output_language=None,
    num_keywords=10,
    topic_nums_to_compare=None,
    corpuses_to_compare=None,
    ignore_words=None,
    min_freq=2,
    min_word_len=3,
    sample_size=1,
    prompt_remove_words=True,
    verbose=True,
    fig_size=(20, 10),
    incl_most_freq=True,
    org_by_pos=True,
    incl_visuals=True,
    zip_results=True,
):
    """
    Generates a directory or zip file of all keyword analysis elements

    Parameters
    ----------
        Most parameters for the following kwx functions:

            utils._prepare_corpus_path

            visuals.graph_topic_num_evals

            visuals.gen_word_cloud

            visuals.pyLDAvis_topics

            model.extract_kws

            utils.prompt_for_word_removal

        incl_most_freq : bool (default=True)
            Whether to include the most frequent words in the output

        org_by_pos : bool (default=True)
            Whether to organize words by their parts of speech

        incl_visuals : str or bool (default=True)
            Which visual graphs to include in the output

            Str options: topic_num_evals, word_cloud, pyLDAvis, t_sne

            Bool options: True - all; False - none

        zip_results : bool (default=True)
            Whether to zip the results from the analysis

    Returns
    -------
        A directory or zip file in the current working directory
    """

    def get_varname(var):
        """
        Returns a variable's name

        Notes
        -----
            For the purpose of converting an object's name to that of the directory or zip

        Parameters
        ----------
            var : obj
                A variable

        Returns
        -------
            name : str
                The variable's name
        """
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()

        name = [var_name for var_name, var_val in callers_local_vars if var_val is var][
            0
        ]

        return name

    if type(text_corpus) == str:
        if os.path.exists(text_corpus):
            dest_name = text_corpus.split("/")[-1].split(".")[0] + "_kws"
        else:
            ValueError("An invalid path was provided for the data location.")

    elif type(text_corpus) == pd.DataFrame or type(text_corpus) == list:
        # Name the output after the text_corpus
        dest_name = get_varname(text_corpus) + "_kws"

    if zip_results:
        dest_name += ".zip"
        if os.path.exists(os.getcwd() + "/" + dest_name):
            os.remove(os.getcwd() + "/" + dest_name)

    else:
        # Create a directory
        dest_name = os.getcwd() + "/" + dest_name
        os.makedirs(dest_name)
        if os.path.exists(dest_name):
            os.rmdir(dest_name)

    # Provide destinations for visuals
    topic_num_evals_dest = False
    word_cloud_dest = False
    pyLDAvis_dest = False
    t_sne_dest = False

    if type(incl_visuals) == str:
        incl_visuals = [incl_visuals]

    if type(incl_visuals) == list:
        if "topic_num_evals" in incl_visuals:
            topic_num_evals_dest = dest_name

        if "word_cloud" in incl_visuals:
            word_cloud_dest = dest_name

        if "pyLDAvis" in incl_visuals:
            pyLDAvis_dest = dest_name

        if "t_sne" in incl_visuals:
            t_sne_dest = dest_name

    else:
        if incl_visuals == True:
            topic_num_evals_dest = dest_name
            word_cloud_dest = dest_name
            pyLDAvis_dest = dest_name
            t_sne_dest = dest_name

    if input_language in languages.lem_abbr_dict().keys():
        input_language = languages.lem_abbr_dict()[input_language]

    if output_language == None:
        output_language = input_language
    else:
        output_language = output_language.lower()
        if output_language in languages.lem_abbr_dict().keys():
            output_language = languages.lem_abbr_dict()[output_language]

    text_corpus, clean_texts = utils._prepare_corpus_path(
        text_corpus=text_corpus,
        clean_texts=clean_texts,
        input_language=input_language,
        min_freq=min_freq,
        min_word_len=min_word_len,
        sample_size=sample_size,
    )

    # Graph metrics and derive the best model and number of topics from them
    (
        best_method,
        model_ideal_topic_num,
        ideal_lda_num_topics,
    ) = visuals.graph_topic_num_evals(
        method=method,
        text_corpus=text_corpus,
        clean_texts=clean_texts,
        input_language=input_language,
        num_keywords=num_keywords,
        topic_nums_to_compare=topic_nums_to_compare,
        min_freq=min_freq,
        min_word_len=min_word_len,
        sample_size=sample_size,
        metrics=True,
        fig_size=fig_size,
        save_file=topic_num_evals_dest,
        return_ideal_metrics=True,
        verbose=verbose,
    )

    if pyLDAvis_dest != False and ideal_lda_num_topics != False:
        visuals.pyLDAvis_topics(
            method="lda",
            text_corpus=text_corpus,
            input_language=input_language,
            num_topics=ideal_lda_num_topics,
            min_freq=min_freq,
            min_word_len=min_word_len,
            sample_size=sample_size,
            save_file=pyLDAvis_dest,
            display_ipython=False,
        )

    # Extract most frequent keywords
    most_fred_kw = extract_kws(
        method="frequency",
        text_corpus=text_corpus,
        clean_texts=clean_texts,
        input_language=input_language,
        output_language=output_language,
        num_keywords=num_keywords,
        num_topics=model_ideal_topic_num,
        corpuses_to_compare=None,
        return_topics=False,
        ignore_words=ignore_words,
        min_freq=min_freq,
        min_word_len=min_word_len,
        sample_size=sample_size,
        prompt_remove_words=False,  # prevent recursion
    )

    # Extract keywords based on the best topic model
    model_kw = extract_kws(
        method=best_method,
        text_corpus=text_corpus,
        clean_texts=clean_texts,
        input_language=input_language,
        output_language=output_language,
        num_keywords=num_keywords,
        num_topics=model_ideal_topic_num,
        corpuses_to_compare=None,
        return_topics=False,
        ignore_words=ignore_words,
        min_freq=min_freq,
        min_word_len=min_word_len,
        sample_size=sample_size,
        prompt_remove_words=False,  # prevent recursion
    )

    if prompt_remove_words:
        # Ask user if words should be ignored, and iterate until no more words should be
        more_words_to_ignore = True
        first_iteration = True
        new_words_to_ignore = ignore_words  # initialize so that it can be added to
        while more_words_to_ignore != False:
            if first_iteration == True:
                print("The most frequent keywords are:\n")
                print(most_fred_kw)
                print("")
                print("The {} keywords are:\n".format(best_method.upper()))
                print(model_kw)
            else:
                print("\n")
                print("The new most frequent keywords are:\n")
                print(most_fred_kw)
                print("")
                print("The new {} keywords are:\n".format(best_method.upper()))
                print(model_kw)

            new_words_to_ignore, words_added = utils.prompt_for_word_removal(
                ignore_words=new_words_to_ignore
            )
            first_iteration = False

            if words_added == True:
                most_fred_kw = extract_kws(
                    method="frequency",
                    text_corpus=text_corpus,
                    clean_texts=clean_texts,
                    input_language=input_language,
                    output_language=output_language,
                    num_keywords=num_keywords,
                    num_topics=model_ideal_topic_num,
                    corpuses_to_compare=None,
                    return_topics=False,
                    ignore_words=new_words_to_ignore,
                    min_freq=min_freq,
                    min_word_len=min_word_len,
                    sample_size=sample_size,
                    prompt_remove_words=False,  # prevent recursion
                )

                model_kw = extract_kws(
                    method=best_method,
                    text_corpus=text_corpus,
                    clean_texts=clean_texts,
                    input_language=input_language,
                    output_language=output_language,
                    num_keywords=num_keywords,
                    num_topics=model_ideal_topic_num,
                    corpuses_to_compare=None,
                    return_topics=False,
                    ignore_words=new_words_to_ignore,
                    min_freq=min_freq,
                    min_word_len=min_word_len,
                    sample_size=sample_size,
                    prompt_remove_words=False,  # prevent recursion
                )

            else:
                more_words_to_ignore = False

    if word_cloud_dest != False:
        # Make a word cloud that doesn't include the words that should be ignored
        visuals.gen_word_cloud(
            text_corpus=text_corpus,
            input_language=input_language,
            ignore_words=new_words_to_ignore,
            min_freq=min_freq,
            min_word_len=min_word_len,
            sample_size=sample_size,
            height=500,
            save_file=word_cloud_dest,
        )

    block_feature = True  # t_sne isn't zipping propertly
    if t_sne_dest != False and block_feature == False:
        visuals.t_sne(
            dimension="both",  # 2d and 3d are also options
            text_corpus=text_corpus,
            num_topics=10,
            remove_3d_outliers=True,
            fig_size=fig_size,
            save_file=t_sne_dest,
        )

    if org_by_pos:
        # Organize words by part of speech and format them for a .txt file output
        most_fred_kw = utils.organize_by_pos(
            outputs=most_fred_kw, output_language=output_language
        )
        model_kw = utils.organize_by_pos(
            outputs=model_kw, output_language=output_language
        )

    keywords_dict = {
        "Most Frequent Keywords": most_fred_kw,
        "{} Keywords".format(best_method.upper()): model_kw,
    }

    def add_to_zip_str(input_obj, new_char):
        """
        Adds characters to a string that will be zipped
        """
        input_obj += new_char
        return input_obj

    def add_to_txt_file(input_obj, new_char):
        """
        Adds characters to a string that will be zipped
        """
        input_obj.write(new_char)
        return input_obj

    if zip_results == True:
        edit_fxn = add_to_zip_str
        input_obj = ""

    else:
        edit_fxn = add_to_txt_file
        txt_file = "keywords.txt"
        input_obj = open(txt_file, "w")

    for model_key, model_val in keywords_dict.items():
        if type(keywords_dict[model_key]) == dict:
            input_obj = edit_fxn(input_obj=input_obj, new_char=str(model_key))
            input_obj = edit_fxn(input_obj=input_obj, new_char="\n\n")
            for pos_key in list(model_val.keys()):
                input_obj = edit_fxn(input_obj=input_obj, new_char=str(pos_key))
                input_obj = edit_fxn(input_obj=input_obj, new_char="\n")
                input_obj = edit_fxn(input_obj=input_obj, new_char="-" * len(pos_key))
                input_obj = edit_fxn(input_obj=input_obj, new_char="\n")
                for pos_word in model_val[pos_key]:
                    input_obj = edit_fxn(input_obj=input_obj, new_char=str(pos_word))
                    input_obj = edit_fxn(input_obj=input_obj, new_char="\n")
                input_obj = edit_fxn(input_obj=input_obj, new_char="\n")

            if model_key != list(keywords_dict.keys())[-1]:
                input_obj = edit_fxn(input_obj=input_obj, new_char="=" * len(model_key))
                input_obj = edit_fxn(input_obj=input_obj, new_char="\n\n")

        elif type(keywords_dict[model_key]) == list:
            input_obj = edit_fxn(input_obj=input_obj, new_char=str(model_key))
            input_obj = edit_fxn(input_obj=input_obj, new_char="\n\n")
            for word in keywords_dict[model_key]:
                input_obj = edit_fxn(input_obj=input_obj, new_char=str(word))
                input_obj = edit_fxn(input_obj=input_obj, new_char="\n")
            input_obj = edit_fxn(input_obj=input_obj, new_char="\n")

            if model_key != list(keywords_dict.keys())[-1]:
                input_obj = edit_fxn(input_obj=input_obj, new_char="=" * len(model_key))
                input_obj = edit_fxn(input_obj=input_obj, new_char="\n\n")

    if zip_results == True:
        with zipfile.ZipFile(dest_name, mode="a") as zf:
            zf.writestr(zinfo_or_arcname="keywords.txt", data=input_obj)
            zf.close()
            print("\n")
            print("Analysis zip folder created in the local directory.")
    else:
        input_obj.close()
        print("\n")
        print("Analysis folder created in the local directory.")
