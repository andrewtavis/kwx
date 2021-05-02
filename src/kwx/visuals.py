"""
visuals
-------

Functions for keyword and topic visualization.

Contents:
    save_vis,
    graph_topic_num_evals,
    gen_word_cloud,
    pyLDAvis_topics,
    t_sne
"""

import inspect
import io
import math
import os
import time
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis

try:
    import pyLDAvis.gensim_models as pyLDAvis_gensim
except ImportError:
    import pyLDAvis.gensim as pyLDAvis_gensim

import seaborn as sns
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from IPython import get_ipython
from IPython.display import display
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from wordcloud import WordCloud

warnings.filterwarnings(action="ignore", message=r"Passing", category=FutureWarning)
from sentence_transformers import SentenceTransformer

from kwx import model, topic_model, utils


def save_vis(vis, save_file, file_name):
    """
    Saves a visualization file in the local or given directory if directed.

    Parameters
    ----------
        vis : matplotlib.pyplot
            The visualization to be saved.

        save_file : bool or str (default=False)
            Whether to save the figure as a png or a path in which to save it.

            Note: directory paths can begin from the working directory.

        file_name : str
            The name for the file.

    Returns
    -------
        The file saved in the local or given directory if directed.
    """
    if save_file == True:
        vis.savefig(
            f"{file_name}_{time.strftime('%Y%m%d-%H%M%S')}.png",
            bbox_inches="tight",
            dpi=300,
        )

    elif isinstance(save_file, str):  # a save path has been provided
        if save_file[-4:] == ".zip":
            with zipfile.ZipFile(save_file, mode="a") as zf:
                vis.plot([0, 0])
                buf = io.BytesIO()
                vis.savefig(buf, bbox_inches="tight", dpi=300)
                vis.close()
                zf.writestr(zinfo_or_arcname=f"{file_name}.png", data=buf.getvalue())
                zf.close()

        else:
            if os.path.exists(save_file):
                vis.savefig(
                    save_file + f"/{file_name}.png", bbox_inches="tight", dpi=300,
                )

            else:
                vis.savefig(
                    f"{file_name}_{time.strftime('%Y%m%d-%H%M%S')}.png",
                    bbox_inches="tight",
                    dpi=300,
                )


def graph_topic_num_evals(
    method=["lda", "bert"],
    bert_st_model="xlm-r-bert-base-nli-stsb-mean-tokens",
    text_corpus=None,
    num_keywords=10,
    topic_nums_to_compare=None,
    metrics=True,
    fig_size=(20, 10),
    save_file=False,
    return_ideal_metrics=False,
    verbose=True,
    **kwargs,
):
    """
    Graphs metrics for the given models over the given number of topics.

    Parameters
    ----------
        method : str (default=["lda", "bert"])
            The modelling method.

            Options:
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

        num_keywords : int (default=10)
            The number of keywords that should be extracted.

        topic_nums_to_compare : list (default=None)
            The number of topics to compare metrics over.

            Note: None selects all numbers from 1 to num_keywords.

        sample_size : float (default=None: sampling for non-BERT techniques)
            The size of a sample for BERT models.

        metrics : str or bool (default=True: all metrics)
            The metrics to include.

            Options:
                - stability: model stability based on Jaccard similarity.

                - coherence: how much the words associated with model topics co-occur.

        fig_size : tuple (default=(20,10))
            The size of the figure.

        save_file : bool or str (default=False)
            Whether to save the figure as a png or a path in which to save it.

        return_ideal_metrics : bool (default=False)
            Whether to return the ideal number of topics for the best model based on metrics.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query.

        **kwargs : keyword arguments
            Keyword arguments correspoding to sentence_transformers.SentenceTransformer.encode or gensim.models.ldamulticore.LdaMulticore.

    Returns
    -------
        ax : matplotlib axis
            A graph of the given metrics for each of the given models based on each topic number.
    """
    assert (
        metrics == "stability" or metrics == "coherence" or metrics == True
    ), "An invalid value has been passed to the 'metrics' argument - please choose from 'stability', 'coherence', or True for both."

    if metrics == True:
        metrics = ["stability", "coherence"]

    if isinstance(method, str):
        method = [method]

    method = [m.lower() for m in method]

    def jaccard_similarity(topic_1, topic_2):
        """
        Derives the Jaccard similarity of two topics.

        Notes
        -----
            Jaccard similarity:
                - A statistic used for comparing the similarity and diversity of sample sets.
                - J(A,B) = (A ∩ B)/(A ∪ B).
                - Goal is low Jaccard scores for coverage of the diverse elements.
        """
        # Fix for cases where there are not enough texts for clustering models.
        if topic_1 == [] and topic_2 != []:
            topic_1 = topic_2
        if topic_1 != [] and topic_2 == []:
            topic_2 = topic_1
        if topic_1 == [] and topic_2 == []:
            topic_1, topic_2 = ["_None"], ["_None"]
        intersection = set(topic_1).intersection(set(topic_2))
        num_intersect = float(len(intersection))

        union = set(topic_1).union(set(topic_2))
        num_union = float(len(union))

        return num_intersect / num_union

    plt.figure(figsize=fig_size)  # begin figure
    metric_vals = []  # add metric values so that figure y-axis can be scaled

    # Initialize the topics numbers that models should be run for.
    if topic_nums_to_compare is None:
        topic_nums_to_compare = list(range(num_keywords + 2))[1:]
    else:
        # If topic numbers are given, then add one more for comparison.
        topic_nums_to_compare = topic_nums_to_compare + [topic_nums_to_compare[-1] + 1]

    ideal_topic_num_dict = {}
    for m in method:
        topics_dict = {}
        stability_dict = {}
        coherence_dict = {}

        bert_model = None
        if m == "bert":
            bert_model = SentenceTransformer(bert_st_model)

        for t_n in tqdm(
            topic_nums_to_compare, desc=f"{m}-topics", disable=not verbose,
        ):
            tm = topic_model.TopicModel(num_topics=t_n, method=m, bert_model=bert_model)
            tm.fit(text_corpus=text_corpus, method=m, m_clustering=None, **kwargs)

            # Assign topics given the current number t_n.
            topics_dict[t_n] = model._order_and_subset_by_coherence(
                tm=tm, num_topics=t_n, num_keywords=num_keywords
            )[0]

            coherence_dict[t_n] = model.get_coherence(
                model=tm,
                text_corpus=text_corpus,
                num_topics=t_n,
                num_keywords=num_keywords,
                measure="c_v",
            )

        if "stability" in metrics:
            for j in range(0, len(topic_nums_to_compare) - 1):
                jaccard_sims = []
                for t1, topic1 in enumerate(  # pylint: disable=unused-variable
                    topics_dict[topic_nums_to_compare[j]]
                ):
                    sims = []
                    for t2, topic2 in enumerate(  # pylint: disable=unused-variable
                        topics_dict[topic_nums_to_compare[j + 1]]
                    ):
                        sims.append(jaccard_similarity(topic1, topic2))

                    jaccard_sims.append(sims)

                stability_dict[topic_nums_to_compare[j]] = np.array(jaccard_sims).mean()

            mean_stabilities = [
                stability_dict[t_n] for t_n in topic_nums_to_compare[:-1]
            ]
            metric_vals += mean_stabilities

            ax = sns.lineplot(
                x=topic_nums_to_compare[:-1],
                y=mean_stabilities,
                label="{}: Average Topic Overlap".format(m.upper()),
            )

        if "coherence" in metrics:
            coherences = [coherence_dict[t_n] for t_n in topic_nums_to_compare[:-1]]
            metric_vals += coherences

            ax = sns.lineplot(
                x=topic_nums_to_compare[:-1],
                y=coherences,
                label="{}: Topic Coherence".format(m.upper()),
            )

        # If both metrics can be calculated, then an optimal number of
        # topics can be derived.
        if "stability" in metrics and "coherence" in metrics:
            coh_sta_diffs = [
                coherences[i] - mean_stabilities[i]
                for i in range(len(topic_nums_to_compare))[:-1]
            ]
            coh_sta_max = max(coh_sta_diffs)
            coh_sta_max_idxs = [
                i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max
            ]
            model_ideal_topic_num_index = coh_sta_max_idxs[
                0
            ]  # take lower topic numbers if more than one max
            model_ideal_topic_num = topic_nums_to_compare[model_ideal_topic_num_index]

            plot_model_ideal_topic_num = model_ideal_topic_num
            if plot_model_ideal_topic_num == topic_nums_to_compare[-1] - 1:
                # Prevents the line from not appearing on the plot.
                plot_model_ideal_topic_num = plot_model_ideal_topic_num - 0.005
            elif plot_model_ideal_topic_num == topic_nums_to_compare[0]:
                # Prevents the line from not appearing on the plot.
                plot_model_ideal_topic_num = plot_model_ideal_topic_num + 0.005

            ax.axvline(
                x=plot_model_ideal_topic_num,
                label="{} Ideal Num Topics: {}".format(
                    m.upper(), model_ideal_topic_num
                ),
                color="black",
            )

            ideal_topic_num_dict[m] = (model_ideal_topic_num, coh_sta_max)

    # Set plot limits.
    y_max = max(metric_vals) + (0.10 * max(metric_vals))
    ax.set_ylim([0, y_max])
    ax.set_xlim([topic_nums_to_compare[0], topic_nums_to_compare[-1] - 1])

    ax.axes.set_title("Method Metrics per Number of Topics", fontsize=25)
    ax.set_ylabel("Metric Level", fontsize=20)
    ax.set_xlabel("Number of Topics", fontsize=20)
    plt.legend(fontsize=20, ncol=len(method))

    # Save file if directed to.
    save_vis(vis=plt, save_file=save_file, file_name="topic_number_metrics")

    # Return the ideal model and its topic number, as well as the best
    # LDA topic number for pyLDAvis.
    if return_ideal_metrics:
        if "lda" in method:
            ideal_lda_num_topics = ideal_topic_num_dict["lda"][0]
        else:
            ideal_lda_num_topics = False

        ideal_topic_num_dict = {
            k: v[0]
            for k, v in sorted(
                ideal_topic_num_dict.items(), key=lambda item: item[1][1]
            )[::-1]
        }
        ideal_model_and_num_topics = next(iter(ideal_topic_num_dict.items()))
        ideal_model, ideal_num_topics = (
            ideal_model_and_num_topics[0],
            ideal_model_and_num_topics[1],
        )

        return ideal_model, ideal_num_topics, ideal_lda_num_topics

    else:
        return ax


def gen_word_cloud(
    text_corpus, ignore_words=None, height=500, save_file=False,
):
    """
    Generates a word cloud for a group of words.

    Parameters
    ----------
        text_corpus : list or list of lists
            The text_corpus that should be plotted.

        ignore_words : str or list (default=None)
            Words that should be removed.

        height : int (default=500)
            The height of the resulting figure
            Note: the width will be the golden ratio times the height.

        save_file : bool or str (default=False)
            Whether to save the figure as a png or a path in which to save it.

    Returns
    -------
        plt.savefig or plt.show : pyplot methods
            A word cloud based on the occurrences of words in a list without removed words.
    """
    flat_words = utils._combine_texts_to_str(
        text_corpus=text_corpus, ignore_words=ignore_words
    )

    width = int(
        height * ((1 + math.sqrt(5)) / 2)
    )  # width is the height multiplied by the golden ratio
    wordcloud = WordCloud(
        width=width, height=height, random_state=None, max_font_size=100
    ).generate(flat_words)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    # Save file if directed to.
    save_vis(vis=plt, save_file=save_file, file_name="word_cloud")

    plt.show()


def pyLDAvis_topics(
    method="lda",
    text_corpus=None,
    num_topics=10,
    save_file=False,
    display_ipython=False,
    **kwargs,
):
    """
    Returns the outputs of an LDA model plotted using pyLDAvis.

    Parameters
    ----------
        method : str or list (default=LDA)
            The modelling method or methods to compare.

            Option:
                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories.

                    - These categories are then used to classify individual entries given the percent they fall into categories.

        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done.

        num_topics : int (default=10)
            The number of categories for LDA and BERT based approaches.

        save_file : bool or str (default=False)
            Whether to save the HTML file to the current working directory or a path in which to save it.

        display_ipython : bool (default=False)
            Whether iPython's display function should be used if in that working environment.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query.

        **kwargs : keyword arguments
            Keyword arguments correspoding to gensim.models.ldamulticore.LdaMulticore.

    Returns
    -------
        pyLDAvis.save_html or pyLDAvis.show : pyLDAvis methods
            A visualization of the topics and their main keywords via pyLDAvis.
    """
    method = method.lower()

    tm = topic_model.TopicModel(num_topics=num_topics, method=method)
    tm.fit(text_corpus=text_corpus, method=method, m_clustering=None, **kwargs)

    def in_ipython():
        """
        Allows for direct display in a Jupyter notebook.
        """
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    vis = pyLDAvis_gensim.prepare(tm.lda_model, tm.bow_corpus, tm.dirichlet_dict)

    if save_file == True:
        pyLDAvis.save_html(
            vis, "lda_topics_{}.html".format(time.strftime("%Y%m%d-%H%M%S"))
        )
    elif isinstance(save_file, str):
        if save_file[-4:] == ".zip":
            pyLDAvis.save_html(vis, "lda_topics.html")
            with zipfile.ZipFile(save_file, mode="a") as zf:
                zf.write(filename="lda_topics.html")
                os.remove("lda_topics.html")
                zf.close()
        else:
            if os.path.exists(save_file):
                pyLDAvis.save_html(vis, save_file + "/lda_topics.html")
            else:
                pyLDAvis.save_html(
                    vis, "/lda_topics_{}.html".format(time.strftime("%Y%m%d-%H%M%S"))
                )

    else:
        if in_ipython() == True and display_ipython == True:
            pyLDAvis.enable_notebook()
            # Display in an ipython notebook.
            display(pyLDAvis.display(vis))
        else:
            # Opens HTML.
            pyLDAvis.show(vis)


def t_sne(
    dimension="both",
    text_corpus=None,
    num_topics=10,
    remove_3d_outliers=False,
    fig_size=(20, 10),
    save_file=False,
    **kwargs,
):
    """
    Returns the outputs of an LDA model plotted using t-SNE (t-distributed Stochastic Neighbor Embedding).

    Notes
    -----
        t-SNE reduces the dimensionality of a space such that similar points will be closer and dissimilar points farther.

    Parameters
    ----------
        dimension : str (default=both)
            The dimension that t-SNE should reduce the data to for visualization
            Options: 2d, 3d, and both (a plot with two subplots).

        text_corpus : list, list of lists
            The tokenized and cleaned text corpus over which analysis should be done.

        num_topics : int (default=10)
            The number of categories for LDA based approaches.

        remove_3d_outliers : bool (default=False)
            Whether to remove outliers from a 3d plot.

        fig_size : tuple (default=(20,10))
            The size of the figure.

        save_file : bool or str (default=False)
            Whether to save the figure as a png or a path in which to save it.

        **kwargs : keyword arguments
            Keyword arguments correspoding to gensim.models.ldamulticore.LdaMulticore or sklearn.manifold.TSNE.

    Returns
    -------
        fig : matplotlib.pyplot.figure
            A t-SNE lower dimensional representation of an LDA model's topics and their constituent members.
    """
    token_corpus = [t.split(" ") for t in text_corpus]
    dirichlet_dict = corpora.Dictionary(token_corpus)
    bow_corpus = [dirichlet_dict.doc2bow(text) for text in token_corpus]

    lda_kwargs = {
        k: v for k, v in kwargs.items() if k in inspect.getfullargspec(LdaMulticore)[0]
    }

    dirichlet_model = LdaMulticore(
        corpus=bow_corpus, id2word=dirichlet_dict, num_topics=num_topics, **lda_kwargs
    )

    df_topic_coherences = pd.DataFrame(
        columns=["topic_{}".format(i) for i in range(num_topics)]
    )

    for i, b in enumerate(bow_corpus):
        df_topic_coherences.loc[i] = [0] * num_topics

        output = dirichlet_model.__getitem__(bow=b, eps=0)

        for o in output:
            topic_num = o[0]
            coherence = o[1]
            df_topic_coherences.iloc[i, topic_num] = coherence

    for i in range(num_topics):
        df_topic_coherences.iloc[:, i] = df_topic_coherences.iloc[:, i].astype(
            "float64", copy=False
        )

    df_topic_coherences["main_topic"] = df_topic_coherences.iloc[:, :num_topics].idxmax(
        axis=1
    )

    if num_topics > 10:
        # cubehelix better for more than 10 colors.
        colors = sns.color_palette("cubehelix", num_topics)
    else:
        # The default sns color palette.
        colors = sns.color_palette("deep", num_topics)

    tsne_2 = None
    tsne_3 = None
    tsne_kwargs = {
        k: v for k, v in kwargs.items() if k in inspect.getfullargspec(TSNE)[0]
    }
    if dimension == "both":
        tsne_2 = TSNE(n_components=2, **tsne_kwargs)
        tsne_3 = TSNE(n_components=3, **tsne_kwargs)

    elif dimension == "2d":
        tsne_2 = TSNE(n_components=2, **tsne_kwargs)

    elif dimension == "3d":
        tsne_3 = TSNE(n_components=3, **tsne_kwargs)
    else:
        ValueError(
            "An invalid value has been passed to the 'dimension' argument - choose from 2d, 3d, or both."
        )

    light_grey_tup = (242 / 256, 242 / 256, 242 / 256)

    if tsne_2 is not None:
        tsne_results_2 = tsne_2.fit_transform(df_topic_coherences.iloc[:, :num_topics])

        df_tsne_2 = pd.DataFrame()
        df_tsne_2["tsne-2d-d1"] = tsne_results_2[:, 0]
        df_tsne_2["tsne-2d-d2"] = tsne_results_2[:, 1]
        df_tsne_2["main_topic"] = df_topic_coherences.iloc[:, num_topics]
        df_tsne_2["color"] = [
            colors[int(t.split("_")[1])] for t in df_tsne_2["main_topic"]
        ]

        df_tsne_2["topic_num"] = [int(i.split("_")[1]) for i in df_tsne_2["main_topic"]]
        df_tsne_2 = df_tsne_2.sort_values(["topic_num"], ascending=True).drop(
            "topic_num", axis=1
        )

    if tsne_3 is not None:
        colors = [c for c in sns.color_palette()]

        tsne_results_3 = tsne_3.fit_transform(df_topic_coherences.iloc[:, :num_topics])

        df_tsne_3 = pd.DataFrame()
        df_tsne_3["tsne-3d-d1"] = tsne_results_3[:, 0]
        df_tsne_3["tsne-3d-d2"] = tsne_results_3[:, 1]
        df_tsne_3["tsne-3d-d3"] = tsne_results_3[:, 2]
        df_tsne_3["main_topic"] = df_topic_coherences.iloc[:, num_topics]
        df_tsne_3["color"] = [
            colors[int(t.split("_")[1])] for t in df_tsne_3["main_topic"]
        ]

        df_tsne_3["topic_num"] = [int(i.split("_")[1]) for i in df_tsne_3["main_topic"]]
        df_tsne_3 = df_tsne_3.sort_values(["topic_num"], ascending=True).drop(
            "topic_num", axis=1
        )

        if remove_3d_outliers:
            # Remove those rows with values that are more than three standard
            # deviations from the column mean.
            for col in ["tsne-3d-d1", "tsne-3d-d2", "tsne-3d-d3"]:
                df_tsne_3 = df_tsne_3[
                    np.abs(df_tsne_3[col] - df_tsne_3[col].mean())
                    <= (3 * df_tsne_3[col].std())
                ]

    if tsne_2 is not None and tsne_3 is not None:
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=fig_size  # pylint: disable=unused-variable
        )
        ax1.axis("off")

    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    if tsne_2 is not None and tsne_3 is not None:
        # Plot tsne_2, with tsne_3 being added later.
        ax1 = sns.scatterplot(
            data=df_tsne_2,
            x="tsne-2d-d1",
            y="tsne-2d-d2",
            hue=df_topic_coherences.iloc[:, num_topics],
            alpha=0.3,
        )

        ax1.set_facecolor(light_grey_tup)
        ax1.axes.set_title("t-SNE 2-Dimensional Representation", fontsize=25)
        ax1.set_xlabel("tsne-d1", fontsize=20)
        ax1.set_ylabel("tsne-d2", fontsize=20)

        handles, labels = ax1.get_legend_handles_labels()
        legend_order = list(np.argsort([i.split("_")[1] for i in labels]))
        ax1.legend(
            [handles[i] for i in legend_order],
            [labels[i] for i in legend_order],
            facecolor=light_grey_tup,
        )

    elif tsne_2 is not None:
        # Plot just tsne_2.
        ax = sns.scatterplot(
            data=df_tsne_2,
            x="tsne-2d-d1",
            y="tsne-2d-d2",
            hue=df_topic_coherences.iloc[:, num_topics],
            alpha=0.3,
        )

        ax.set_facecolor(light_grey_tup)
        ax.axes.set_title("t-SNE 2-Dimensional Representation", fontsize=25)
        ax.set_xlabel("tsne-d1", fontsize=20)
        ax.set_ylabel("tsne-d2", fontsize=20)

        handles, labels = ax.get_legend_handles_labels()
        legend_order = list(np.argsort([i.split("_")[1] for i in labels]))
        ax.legend(
            [handles[i] for i in legend_order],
            [labels[i] for i in legend_order],
            facecolor=light_grey_tup,
        )

    if tsne_2 is not None and tsne_3 is not None:
        # tsne_2 has been plotted, so add tsne_3.
        ax2 = fig.add_subplot(121, projection="3d")
        ax2.scatter(
            xs=df_tsne_3["tsne-3d-d1"],
            ys=df_tsne_3["tsne-3d-d2"],
            zs=df_tsne_3["tsne-3d-d3"],
            c=df_tsne_3["color"],
            alpha=0.3,
        )

        ax2.set_facecolor("white")
        ax2.axes.set_title("t-SNE 3-Dimensional Representation", fontsize=25)
        ax2.set_xlabel("tsne-d1", fontsize=20)
        ax2.set_ylabel("tsne-d2", fontsize=20)
        ax2.set_zlabel("tsne-d3", fontsize=20)

        with plt.rc_context({"lines.markeredgewidth": 0}):
            # Add handles via blank lines and order their colors to match tsne_2.
            proxy_handles = [
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    markersize=8,
                    markerfacecolor=colors[i],
                )
                for i in legend_order
            ]
            ax2.legend(
                proxy_handles,
                ["topic_{}".format(i) for i in range(num_topics)],
                loc="upper left",
                facecolor=(light_grey_tup),
            )

    elif tsne_3 is not None:
        # Plot just tsne_3.
        ax.axis("off")
        ax.set_facecolor("white")
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            xs=df_tsne_3["tsne-3d-d1"],
            ys=df_tsne_3["tsne-3d-d2"],
            zs=df_tsne_3["tsne-3d-d3"],
            c=df_tsne_3["color"],
            alpha=0.3,
        )

        ax.set_facecolor("white")
        ax.axes.set_title("t-SNE 3-Dimensional Representation", fontsize=25)
        ax.set_xlabel("tsne-d1", fontsize=20)
        ax.set_ylabel("tsne-d2", fontsize=20)
        ax.set_zlabel("tsne-d3", fontsize=20)

        with plt.rc_context({"lines.markeredgewidth": 0}):
            # Add handles via blank lines.
            proxy_handles = [
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    markersize=8,
                    markerfacecolor=c,
                )
                for i, c in enumerate(colors)
            ]
            ax.legend(
                proxy_handles,
                ["topic_{}".format(i) for i in range(num_topics)],
                loc="upper left",
                facecolor=light_grey_tup,
            )

    # Save file if directed to.
    save_vis(vis=plt, save_file=save_file, file_name="t_sne")

    return fig
