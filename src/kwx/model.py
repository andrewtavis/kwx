"""
model
-----

Functions for modeling text corpuses and extracting keywords.

Contents:
    get_topic_words,
    get_coherence,
    _order_and_subset_by_coherence,
    _select_kws,
    extract_kws,
    gen_files
"""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from collections import Counter

import math
import os
import time
import zipfile

import numpy as np
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings(action="ignore", message=r"Passing", category=FutureWarning)
from sentence_transformers import SentenceTransformer

from kwx import languages, topic_model, utils, visuals


def get_topic_words(text_corpus, labels, num_topics=None, num_keywords=None):
    """
    Get top words within each topic for cluster models.

    Parameters
    ----------
        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done.

        labels : list
            The labels assigned to topics.

        num_topics : int (default=None)
            The number of categories for LDA and BERT based approaches.

        num_keywords : int (default=None)
            The number of keywords that should be extracted.

    Returns
    -------
        topics, non_blank_topic_idxs : list and list
            Topic keywords and indexes of those that are not empty lists.
    """
    if num_topics is None:
        num_topics = len(np.unique(labels))
    topics = ["" for _ in range(num_topics)]

    for i, c in enumerate(text_corpus):
        topics[labels[i]] += " " + "".join(c)

    # Count the words that appear for a given topic label.
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
    Gets model coherence from gensim.models.coherencemodel.

    Parameters
    ----------
        model : kwx.topic_model.TopicModel
            A model trained on the given text corpus.

        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done.

        num_topics : int (default=10)
            The number of categories for LDA and BERT based approaches.

        num_keywords : int (default=10)
            The number of keywords that should be extracted.

        measure : str (default=c_v)
            A gensim measure of coherence.

    Returns
    -------
        coherence : float
            The coherence of the given model over the given texts.
    """
    token_corpus = [t.split(" ") for t in text_corpus]

    if model.method.lower() == "lda":
        cm = CoherenceModel(
            model=model.lda_model,
            texts=token_corpus,
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
            texts=token_corpus,
            corpus=model.bow_corpus,
            dictionary=model.dirichlet_dict,
            coherence=measure,
        )

    return cm.get_coherence()


def _order_and_subset_by_coherence(tm, num_topics=10, num_keywords=10):
    """
    Orders topics based on their average coherence across the text corpus.

    Parameters
    ----------
        tm : kwx.topic_model.TopicModel
            A model trained on the given text corpus.

        num_topics : int (default=10)
            The number of categories for LDA and BERT based approaches.

        num_keywords : int (default=10)
            The number of keywords that should be extracted.

    Returns
    -------
        ordered_topic_words, selection_indexes: list of lists and list of lists
            Topics words ordered by average coherence and indexes by which they should be selected.
    """
    # Derive average topics across texts for a given method
    if tm.method == "lda":
        shown_topics = tm.lda_model.show_topics(
            num_topics=num_topics, num_words=num_keywords, formatted=False
        )

        topic_words = [[word[0] for word in topic[1]] for topic in shown_topics]
        topic_corpus = tm.lda_model.__getitem__(
            bow=tm.bow_corpus, eps=0
        )  # cutoff probability to 0

        topics_per_response = [response for response in topic_corpus]
        flat_topic_coherences = [
            item for sublist in topics_per_response for item in sublist
        ]

        topic_averages = [
            (
                t,
                sum(t_c[1] for t_c in flat_topic_coherences if t_c[0] == t)
                / len(tm.bow_corpus),
            )
            for t in range(num_topics)
        ]

    elif tm.method == "bert":
        # The topics in cluster models are not guranteed to be the size of num_keywords.
        topic_words, non_blank_topic_idxs = get_topic_words(
            text_corpus=tm.text_corpus,
            labels=tm.cluster_model.labels_,
            num_topics=num_topics,
            num_keywords=num_keywords,
        )

        # Create a dictionary of the assignment counts for the topics.
        counts_dict = dict(Counter(tm.cluster_model.labels_))
        counts_dict = {
            k: v for k, v in counts_dict.items() if k in non_blank_topic_idxs
        }
        keys_ordered = sorted([k for k in counts_dict])

        # Map to the range from 0 to the number of non-blank topics.
        counts_dict_mapped = {i: counts_dict[k] for i, k in enumerate(keys_ordered)}

        # Derive the average assignment of the topics.
        topic_averages = [
            (k, counts_dict_mapped[k] / sum(counts_dict_mapped.values()))
            for k in counts_dict_mapped
        ]

    # Order ids by the average coherence across the texts.
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

    # Create selection indexes for each topic given its average coherence
    # and how many keywords are wanted.
    selection_indexes = [
        list(range(int(math.floor(num_keywords * a))))
        if math.floor(num_keywords * a) > 0
        else [0]
        for i, a in enumerate(ordered_topic_averages)
    ]

    total_indexes = sum(len(i) for i in selection_indexes)
    s_i = 0
    while total_indexes < num_keywords:
        selection_indexes[s_i] = selection_indexes[s_i] + [
            selection_indexes[s_i][-1] + 1
        ]
        s_i += 1
        total_indexes += 1

    return ordered_topic_words, selection_indexes


def _select_kws(method="lda", kw_args=None, words_to_ignore=None, n=10):
    """
    Selects keywords from a group of extracted keywords.

    Parameters
    ----------
        method : str (default=lda)
            The modelling method.

            Options:
                frequency: a count of the most frequent words.

                TFIDF: Term Frequency Inverse Document Frequency.

                    - Allows for words within one text group to be compared to those of another.
                    - Gives a better idea of what users specifically want from a given publication.

                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories.
                    - These categories are then used to classify individual entries given the percent they fall into categories.

                BERT: Bidirectional Encoder Representations from Transformers

                    - Words are classified via Google Neural Networks.
                    - Word classifications are then used to derive topics.

        kw_args : dict (default=None)
            A dictionary of keywords and metrics through which to order them as values.

        words_to_ignore : list (default=None)
            Words to not include in the selected keywords.

        n : int (default=10)
            The number of keywords to select.

    Returns
    -------
        keywords : list
            Selected keywords from those extracted.
    """
    if method in ["frequency", "tfidf"]:
        kw_dict = {
            k: v
            for k, v in sorted(kw_args.items(), key=lambda item: item[1])[::-1]
            if k not in words_to_ignore
        }

        keywords = list(kw_dict.keys())[:n]

    elif method in ["lda", "bert"]:
        ordered_topic_words, selection_indexes = kw_args

        # Reverse all selection variables so that low level words come from strong topics.
        ordered_topic_words = ordered_topic_words[::-1]
        selection_indexes = selection_indexes[::-1]

        flat_ordered_topic_words = [
            word for topic in ordered_topic_words for word in topic
        ]
        set_ordered_topic_words = list(set(flat_ordered_topic_words))
        set_ordered_topic_words = [
            t_w for t_w in set_ordered_topic_words if t_w not in words_to_ignore
        ]
        if len(set_ordered_topic_words) <= n:
            print("\n")
            print(
                "The number of distinct topic words is less than the desired number of keywords."
            )
            print("All topic words will be returned.")
            keywords = set_ordered_topic_words

        else:
            # Derive keywords from Dirichlet or cluster algorithms.
            t_n = 0
            keywords = []
            while len(keywords) < n:
                sel_idxs = selection_indexes[t_n]

                for s_i in sel_idxs:
                    if (
                        ordered_topic_words[t_n][s_i] not in keywords
                        and ordered_topic_words[t_n][s_i] not in words_to_ignore
                    ):
                        keywords.append(ordered_topic_words[t_n][s_i])
                    else:
                        sel_idxs.append(sel_idxs[-1] + 1)

                    if len(sel_idxs) >= len(ordered_topic_words[t_n]):
                        # The indexes are now more than the keywords, so move to
                        # the next topic.
                        break

                t_n += 1
                if t_n == len(ordered_topic_words):
                    # The last topic has been gone through, so return to the first.
                    t_n = 0

        # Fix for if too many were selected.
        keywords = keywords[:n]

    return keywords


def extract_kws(
    method="lda",
    bert_st_model="xlm-r-bert-base-nli-stsb-mean-tokens",
    text_corpus=None,
    input_language=None,
    output_language=None,
    num_keywords=10,
    num_topics=10,
    corpuses_to_compare=None,
    return_topics=False,
    ignore_words=None,
    prompt_remove_words=True,
    return_kw_args=False,
    **kwargs,
):
    """
    Extracts keywords given data, metadata, and model parameter inputs.

    Parameters
    ----------
        method : str (default=lda)
            The modelling method.

            Options:
                frequency: a count of the most frequent words.

                TFIDF: Term Frequency Inverse Document Frequency.

                    - Allows for words within one text group to be compared to those of another.
                    - Gives a better idea of what users specifically want from a given publication.

                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories.
                    - These categories are then used to classify individual entries given the percent they fall into categories.

                BERT: Bidirectional Encoder Representations from Transformers

                    - Words are classified via Google Neural Networks.
                    - Word classifications are then used to derive topics.

        bert_st_model : str (deafault=xlm-r-bert-base-nli-stsb-mean-tokens)
            The BERT model to use.

        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done.

        input_language : str (default=None)
            The spoken language in which the texts are found.

        output_language : str (default=None: same as input_language)
            The spoken language in which the results should be given.

        num_keywords : int (default=10)
            The number of keywords that should be extracted.

        num_topics : int (default=10)
            The number of categories for LDA and BERT based approaches.

        corpuses_to_compare : list : contains lists (default=None)
            A list of other text corpuses that the main corpus should be compared to using TFIDF.

        return_topics : bool (default=False)
            Whether to return the topics that are extracted by an LDA model.

        ignore_words : str or list (default=None)
            Words that should be removed.

        prompt_remove_words : bool (default=True)
            Whether to prompt the user for keywords to remove.

        **kwargs : keyword arguments
            Keyword arguments correspoding to sentence_transformers.SentenceTransformer.encode, gensim.models.ldamulticore.LdaMulticore, or sklearn.feature_extraction.text.TfidfVectorizer.

    Returns
    -------
        output_keywords : list or list of lists
            A list of lists where sub_lists are the keywords best associated with the data entry.
    """
    input_language = input_language.lower()
    method = method.lower()

    valid_methods = ["frequency", "tfidf", "lda", "bert"]

    assert method in valid_methods, (
        "The value for the 'method' argument is invalid. Please choose one of "
        + " ".join(m for m in valid_methods)
        + "."
    )

    if method.lower() == "tfidf":
        assert (
            corpuses_to_compare != None
        ), "TFIDF requires another text corpus to be passed to the `corpuses_to_compare` argument."

    if input_language in languages.lem_abbr_dict():
        input_language = languages.lem_abbr_dict()[input_language]

    if output_language is None:
        output_language = input_language
    else:
        output_language = output_language.lower()
        if output_language in languages.lem_abbr_dict():
            output_language = languages.lem_abbr_dict()[output_language]

    if ignore_words is not None:
        if isinstance(ignore_words, str):
            words_to_ignore = [ignore_words]

        elif isinstance(ignore_words, list):
            words_to_ignore = ignore_words

    else:
        words_to_ignore = []

    if method == "frequency" or method == "tfidf":
        if method == "frequency":
            kw_args = Counter(
                item for subtext in text_corpus for item in subtext.split()
            )

            # Return for gen_files.
            if return_kw_args:
                return kw_args

            keywords = _select_kws(
                method=method,
                kw_args=kw_args,
                words_to_ignore=words_to_ignore,
                n=num_keywords,
            )

        elif method == "tfidf":  # Term Frequency Inverse Document Frequency
            if isinstance(corpuses_to_compare[0], str):  # only one corpus to compare
                corpuses_to_compare = [corpuses_to_compare]

            # Combine the main corpus and those to compare.
            comparative_corpus = [corpuses_to_compare]
            comparative_corpus.insert(0, text_corpus)

            comparative_string_corpus = []
            for c in comparative_corpus:
                combined_tokens = utils._combine_texts_to_str(text_corpus=c)

                comparative_string_corpus.append(combined_tokens)

            tfidf_vectorizer = TfidfVectorizer(**kwargs)
            tm = tfidf_vectorizer.fit_transform(  # pylint: disable=unused-variable
                comparative_string_corpus
            )
            corpus_scored = tfidf_vectorizer.transform(comparative_string_corpus)
            terms = tfidf_vectorizer.get_feature_names()
            scores = corpus_scored.toarray().flatten().tolist()
            kw_args = dict(zip(terms, scores))

            # Return for gen_files.
            if return_kw_args:
                return kw_args

            keywords = _select_kws(
                method=method,
                kw_args=kw_args,
                words_to_ignore=words_to_ignore,
                n=num_keywords,
            )

            # Check that more words than the number that appear in the text is not given.
            frequent_words = extract_kws(
                method="frequency",
                text_corpus=text_corpus,
                input_language=input_language,
                output_language=output_language,
                num_keywords=num_keywords,
                num_topics=num_topics,
                corpuses_to_compare=corpuses_to_compare,
                return_topics=False,
                ignore_words=words_to_ignore,
                prompt_remove_words=False,  # prevent recursion
            )

            if len(keywords) > len(frequent_words):
                keywords = keywords[: len(frequent_words)]

    elif method in ["lda", "bert"]:
        bert_model = None
        if method == "bert":
            bert_model = SentenceTransformer(bert_st_model)

        tm = topic_model.TopicModel(
            num_topics=num_topics, method=method, bert_model=bert_model
        )
        tm.fit(text_corpus=text_corpus, method=method, m_clustering=None)

        ordered_topic_words, selection_indexes = _order_and_subset_by_coherence(
            tm=tm, num_topics=num_topics, num_keywords=num_keywords
        )

        if return_topics:
            if output_language != input_language:
                ordered_topic_words = utils.translate_output(
                    outputs=ordered_topic_words,
                    input_language=input_language,
                    output_language=output_language,
                )

            return ordered_topic_words

        else:
            kw_args = (ordered_topic_words, selection_indexes)

            # Return for gen_files.
            if return_kw_args:
                return kw_args

            keywords = _select_kws(
                method=method,
                kw_args=kw_args,
                words_to_ignore=words_to_ignore,
                n=num_keywords,
            )

            # If there are not enough words, then add non-included most
            # frequent ones in order.
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
                    ignore_words=words_to_ignore,
                    prompt_remove_words=False,  # prevent recursion
                )

                for word in frequent_words:
                    if word not in keywords and len(keywords) < len(frequent_words):
                        keywords.append(word)

    if prompt_remove_words:
        # Ask user if words should be ignored, and iterate until no more words should be.
        more_words_to_ignore = True
        first_iteration = True
        new_words_to_ignore = words_to_ignore  # initialize so that it can be added to
        while more_words_to_ignore != False:
            if first_iteration == True:
                print("The {} keywords are:\n".format(method.upper()))
                print(keywords)

            else:
                print("\n")
                print("The new {} keywords are:\n".format(method.upper()))
                print(keywords)

            new_words_to_ignore, words_added = utils.prompt_for_word_removal(
                words_to_ignore=new_words_to_ignore
            )
            first_iteration = False

            if words_added == True:
                keywords = _select_kws(
                    method=method,
                    kw_args=kw_args,
                    words_to_ignore=new_words_to_ignore,
                    n=num_keywords,
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
    method=["lda", "bert"],
    text_corpus=None,
    input_language=None,
    output_language=None,
    num_keywords=10,
    topic_nums_to_compare=None,
    corpuses_to_compare=None,
    ignore_words=None,
    prompt_remove_words=True,
    verbose=True,
    fig_size=(20, 10),
    incl_most_freq=True,
    org_by_pos=True,
    incl_visuals=True,
    save_dir=None,
    zip_results=True,
):
    """
    Generates a directory or zip file of all keyword analysis elements.

    Parameters
    ----------
        Most parameters for the following kwx functions:

            visuals.graph_topic_num_evals

            visuals.gen_word_cloud

            visuals.pyLDAvis_topics

            model.extract_kws

            utils.prompt_for_word_removal

        incl_most_freq : bool (default=True)
            Whether to include the most frequent words in the output.

        org_by_pos : bool (default=True)
            Whether to organize words by their parts of speech.

        incl_visuals : str or bool (default=True)
            Which visual graphs to include in the output.

            Str options: topic_num_evals, word_cloud, pyLDAvis, t_sne.

            Bool options: True - all; False - none.

        save_dir : str (default=None)
            A path to a directory where the results will be saved.

        zip_results : bool (default=True)
            Whether to zip the results from the analysis.

    Returns
    -------
        A directory or zip file in the current working or save_dir directory.
    """
    if isinstance(method, list):
        if len(method) == 1:
            method = method[0]

    if save_dir is None:
        save_dir = f'keyword_extraction_{time.strftime("%Y%m%d-%H%M%S")}'

    if zip_results:
        if save_dir[-4:] != ".zip":
            save_dir += ".zip"

        if os.path.exists(os.getcwd() + "/" + save_dir):
            os.remove(os.getcwd() + "/" + save_dir)

    else:
        # Create the directory
        save_dir = os.getcwd() + "/" + save_dir
        os.makedirs(save_dir)
        if os.path.exists(save_dir):
            os.rmdir(save_dir)

    # Provide destinations for visuals
    topic_num_evals_dest = False
    word_cloud_dest = False
    pyLDAvis_dest = False
    t_sne_dest = False

    if isinstance(incl_visuals, str):
        incl_visuals = [incl_visuals]

    if isinstance(incl_visuals, list):
        if "topic_num_evals" in incl_visuals:
            topic_num_evals_dest = save_dir

        if "word_cloud" in incl_visuals:
            word_cloud_dest = save_dir

        if "pyLDAvis" in incl_visuals:
            pyLDAvis_dest = save_dir

        if "t_sne" in incl_visuals:
            t_sne_dest = save_dir

    else:
        if incl_visuals == True:
            topic_num_evals_dest = save_dir
            word_cloud_dest = save_dir
            pyLDAvis_dest = save_dir
            t_sne_dest = save_dir

    if input_language in languages.lem_abbr_dict():
        input_language = languages.lem_abbr_dict()[input_language]

    if output_language is None:
        output_language = input_language

    else:
        output_language = output_language.lower()
        if output_language in languages.lem_abbr_dict():
            output_language = languages.lem_abbr_dict()[output_language]

    if ignore_words is not None:
        if isinstance(ignore_words, str):
            words_to_ignore = [ignore_words]

        elif isinstance(ignore_words, list):
            words_to_ignore = ignore_words

    else:
        words_to_ignore = []

    # Graph metrics and derive the best model and number of topics from them.
    (
        best_method,
        model_ideal_topic_num,
        ideal_lda_num_topics,
    ) = visuals.graph_topic_num_evals(
        method=method,
        text_corpus=text_corpus,
        num_keywords=num_keywords,
        topic_nums_to_compare=topic_nums_to_compare,
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
            num_topics=ideal_lda_num_topics,
            save_file=pyLDAvis_dest,
            display_ipython=False,
        )

    # Extract most frequent keywords
    most_freq_kw_args = extract_kws(
        method="frequency",
        text_corpus=text_corpus,
        input_language=input_language,
        output_language=output_language,
        num_keywords=num_keywords,
        num_topics=model_ideal_topic_num,
        corpuses_to_compare=None,
        return_topics=False,
        ignore_words=words_to_ignore,
        prompt_remove_words=False,  # prevent recursion
        return_kw_args=True,
    )

    # Extract keywords based on the best topic model.
    model_kw_args = extract_kws(
        method=best_method,
        text_corpus=text_corpus,
        input_language=input_language,
        output_language=output_language,
        num_keywords=num_keywords,
        num_topics=model_ideal_topic_num,
        corpuses_to_compare=None,
        return_topics=False,
        ignore_words=words_to_ignore,
        prompt_remove_words=False,  # prevent recursion
        return_kw_args=True,
    )

    most_freq_kw = _select_kws(
        method="frequency",
        kw_args=most_freq_kw_args,
        words_to_ignore=words_to_ignore,
        n=num_keywords,
    )

    model_kw = _select_kws(
        method=method,
        kw_args=model_kw_args,
        words_to_ignore=words_to_ignore,
        n=num_keywords,
    )

    if prompt_remove_words:
        # Ask user if words should be ignored, and iterate until no
        # more words should be.
        more_words_to_ignore = True
        first_iteration = True
        new_words_to_ignore = words_to_ignore  # initialize so that it can be added to

        while more_words_to_ignore != False:
            if first_iteration == True:
                print("The most frequent keywords are:\n")
                print(most_freq_kw)
                print("")
                print("The {} keywords are:\n".format(best_method.upper()))
                print(model_kw)

            else:
                print("\n")
                print("The new most frequent keywords are:\n")
                print(most_freq_kw)
                print("")
                print("The new {} keywords are:\n".format(best_method.upper()))
                print(model_kw)

            new_words_to_ignore, words_added = utils.prompt_for_word_removal(
                words_to_ignore=new_words_to_ignore
            )
            first_iteration = False

            if words_added == True:
                most_freq_kw = _select_kws(
                    method="frequency",
                    kw_args=most_freq_kw_args,
                    words_to_ignore=new_words_to_ignore,
                    n=num_keywords,
                )

                model_kw = _select_kws(
                    method=method,
                    kw_args=model_kw_args,
                    words_to_ignore=new_words_to_ignore,
                    n=num_keywords,
                )

            else:
                more_words_to_ignore = False

    if word_cloud_dest != False:
        # Make a word cloud that doesn't include the words that should be ignored.
        visuals.gen_word_cloud(
            text_corpus=text_corpus,
            ignore_words=words_to_ignore,
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
        # Organize words by part of speech and format them for a .txt file output.
        most_freq_kw = utils.organize_by_pos(
            outputs=most_freq_kw, output_language=output_language
        )
        model_kw = utils.organize_by_pos(
            outputs=model_kw, output_language=output_language
        )

    keywords_dict = {
        "Most Frequent Keywords": most_freq_kw,
        "{} Keywords".format(best_method.upper()): model_kw,
    }

    def add_to_zip_str(input_obj, new_char):
        """
        Adds characters to a string that will be zipped.
        """
        input_obj += new_char
        return input_obj

    def add_to_txt_file(input_obj, new_char):
        """
        Adds characters to a string that will be zipped.
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
        if isinstance(keywords_dict[model_key], dict):
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

        elif isinstance(keywords_dict[model_key], list):
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
        with zipfile.ZipFile(save_dir, mode="a") as zf:
            zf.writestr(zinfo_or_arcname="keywords.txt", data=input_obj)
            zf.close()
            print("\n")
            print(f"Analysis zip folder {save_dir} created in the local directory.")

    else:
        input_obj.close()
        print("\n")
        print(f"Analysis folder {save_dir} created in the local directory.")
