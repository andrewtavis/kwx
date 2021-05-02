"""
topic_model
-----------

The unsupervised learning topic model for keyword extraction.

Contents:
    TopicModel Class:
        _vectorize,
        fit
"""

import inspect
import logging
import os
import warnings
from datetime import datetime

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from sklearn.cluster import KMeans


class TopicModel:
    """
    The topic model class to fit and predict given an unsupervised learning technique.
    """

    def __init__(self, num_topics=10, method="lda", bert_model=None):
        """
        Parameters
        ----------
            num_topics : int (default=10)
                The number of categories for LDA and BERT based approaches.

            method : str (default=lda)
                The modelling method.

            bert_model : sentence_transformers.SentenceTransformer.SentenceTransformer
                A sentence transformer model.
        """
        modeling_methods = ["lda", "bert"]
        if method not in modeling_methods:
            ValueError(
                "The indicated method is invalid. Please choose from {}.".format(
                    modeling_methods
                )
            )

        self.num_topics = num_topics
        self.bert_model = bert_model
        self.dirichlet_dict = None
        self.bow_corpus = None
        self.text_corpus = None
        self.cluster_model = None
        self.lda_model = None
        self.vec = {}
        self.gamma = 15  # parameter for relative importance of LDA
        self.method = method.lower()
        self.id = method + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def _vectorize(self, text_corpus, method=None, **kwargs):
        """
        Get vector representations from selected methods.

        Parameters
        ----------
            text_corpus : list, list of lists, or str
                The text corpus over which analysis should be done.

            method : str
                The modeling technique to use.

            **kwargs : keyword arguments
                Keyword arguments correspoding to sentence_transformers.SentenceTransformer.encode or gensim.models.ldamulticore.LdaMulticore.

        Returns
        -------
            vec : np.array
                An array of text vectorizations.
        """
        if method is None:
            method = self.method

        self.text_corpus = text_corpus
        token_corpus = [t.split(" ") for t in text_corpus]
        self.dirichlet_dict = corpora.Dictionary(token_corpus)
        self.bow_corpus = [self.dirichlet_dict.doc2bow(text) for text in token_corpus]

        if method == "lda":
            if not self.lda_model:
                kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k in inspect.getfullargspec(LdaMulticore)[0]
                }
                self.lda_model = LdaMulticore(
                    corpus=self.bow_corpus,
                    num_topics=self.num_topics,
                    id2word=self.dirichlet_dict,
                    **kwargs,
                )

            def get_vec_lda(model, bow_corpus, num_topics):
                """
                Get the LDA vector representation.

                Parameters
                ----------
                    bow_corpus : list of lists
                        Contains doc2bow representations of the given texts.

                    num_topics : int
                        The number of categories for LDA and BERT based approaches.

                Returns
                -------
                    vec_lda : np.array (n_doc * n_topic)
                        The probabilistic topic assignments for all documents.
                """
                n_doc = len(bow_corpus)
                vec_lda = np.zeros((n_doc, num_topics))
                for i in range(n_doc):
                    # Get the distribution for the i-th document in bow_corpus.
                    for topic, prob in model.get_document_topics(
                        bow=bow_corpus[i], minimum_probability=0
                    ):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.lda_model, self.bow_corpus, self.num_topics)

            return vec

        elif method == "bert":
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in inspect.getfullargspec(self.bert_model.encode)[0]
            }
            vec = np.array(self.bert_model.encode(sentences=self.text_corpus, **kwargs))

            return vec

    def fit(self, text_corpus, method=None, m_clustering=None, **kwargs):
        """
        Fit the topic model for selected method given the preprocessed data.

        Parameters
        ----------
            text_corpus : list, list of lists, or str
                The text corpus over which analysis should be done.

            method : str
                The modeling technique to use.

            m_clustering : sklearn.cluster.object
                The method that should be used to cluster.

            **kwargs : keyword arguments
                Keyword arguments correspoding to sentence_transformers.SentenceTransformer.encode or gensim.models.ldamulticore.LdaMulticore.

        Returns
        -------
            self : LDA or cluster model
                A fitted model.
        """
        if method is None:
            method = self.method

        if m_clustering is None:
            m_clustering = KMeans

        self.text_corpus = text_corpus
        if not self.dirichlet_dict:
            token_corpus = [t.split(" ") for t in text_corpus]
            self.dirichlet_dict = corpora.Dictionary(token_corpus)
            self.bow_corpus = [
                self.dirichlet_dict.doc2bow(text) for text in token_corpus
            ]

        if method == "lda":
            if not self.lda_model:
                kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k in inspect.getfullargspec(LdaMulticore)[0]
                }
                self.lda_model = LdaMulticore(
                    corpus=self.bow_corpus,
                    num_topics=self.num_topics,
                    id2word=self.dirichlet_dict,
                    **kwargs,
                )

        else:
            if len(self.text_corpus) < self.num_topics:
                raise ValueError(
                    "`num_topics` cannot be larger than the size of `text_corpus` - consider lowering the desired number of topics"
                )

            self.cluster_model = m_clustering(self.num_topics)
            self.vec[method] = self._vectorize(
                text_corpus=self.text_corpus, method=method, **kwargs,
            )
            self.cluster_model.fit(X=self.vec[method])
