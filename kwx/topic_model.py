"""
topic_model
-----------

The unsupervised learning topic model for keyword extraction

Contents:
    TopicModel Class:
        _vectorize,
        fit,
        predict
"""

from datetime import datetime
import numpy as np

from gensim import corpora
from gensim.models import LdaModel
from sklearn.cluster import KMeans

from kwx.autoencoder import Autoencoder


class TopicModel:
    """
    The topic model class to fit and predict given an unsupervised learning technique
    """

    def __init__(self, num_topics=10, method="lda_bert", bert_model=None):
        """
        Parameters
        ----------
            num_topics : int (default=10)
                The number of categories for LDA and BERT based approaches

            method : str (default=lda_bert)
                The modelling method

            bert_model : sentence_transformers.SentenceTransformer.SentenceTransformer
                A sentence transformer model
        """
        modeling_methods = ["lda", "bert", "lda_bert"]
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
        self.autoencoder = None
        self.id = method + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def _vectorize(self, texts, text_corpus, method=None):
        """
        Get vector representations from selected methods

        Parameters
        ----------
            texts : list
                Text strings that are formatted for cluster models

            text_corpus : list, list of lists, or str
                The text corpus over which analysis should be done

            method : str
                The modeling technique to use

        Returns
        -------
            vec : np.array
                An array of text vectorizations
        """
        if method == None:
            method = self.method

        self.text_corpus = text_corpus
        self.dirichlet_dict = corpora.Dictionary(text_corpus)
        self.bow_corpus = [self.dirichlet_dict.doc2bow(text) for text in text_corpus]

        if method == "lda":
            if not self.lda_model:
                self.lda_model = LdaModel(
                    corpus=self.bow_corpus,
                    num_topics=self.num_topics,
                    id2word=self.dirichlet_dict,
                    chunksize=len(self.bow_corpus),
                    passes=20,  # increase to run model more iterations
                    alpha="auto",
                    random_state=None,
                )

            def get_vec_lda(model, bow_corpus, num_topics):
                """
                Get the LDA vector representation

                Parameters
                ----------
                    bow_corpus : list of lists
                        Contains doc2bow representations of the given texts

                    num_topics : int
                        The number of categories for LDA and BERT based approaches

                Returns
                -------
                    vec_lda : np.array (n_doc * n_topic)
                        The probabilistic topic assignments for all documents
                """
                n_doc = len(bow_corpus)
                vec_lda = np.zeros((n_doc, num_topics))
                for i in range(n_doc):
                    # Get the distribution for the i-th document in bow_corpus
                    for topic, prob in model.get_document_topics(
                        bow=bow_corpus[i], minimum_probability=0
                    ):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.lda_model, self.bow_corpus, self.num_topics)

            return vec

        elif method == "bert":
            model = self.bert_model
            vec = np.array(model.encode(texts, show_progress_bar=False))

            return vec

        elif method == "lda_bert":
            vec_lda = self._vectorize(
                texts=texts, text_corpus=text_corpus, method="lda"
            )
            vec_bert = self._vectorize(
                texts=texts, text_corpus=text_corpus, method="bert"
            )
            vec_bert = vec_bert[: len(vec_lda)]  # Fix if BERT vector larger than LDA's
            vec_lda_bert = np.c_[vec_lda * self.gamma, vec_bert]
            self.vec["LDA_BERT_FULL"] = vec_lda_bert
            if not self.autoencoder:
                self.autoencoder = Autoencoder()
                self.autoencoder.fit(vec_lda_bert)

            vec = self.autoencoder.encoder.predict(vec_lda_bert)

            return vec

    def fit(self, texts, text_corpus, method=None, m_clustering=None):
        """
        Fit the topic model for selected method given the preprocessed data

        Parameters
        ----------
            texts : list
                Text strings that are formatted for cluster models

            text_corpus : list, list of lists, or str
                The text corpus over which analysis should be done

            method : str
                The modeling technique to use

            m_clustering : sklearn.cluster.object
                The method that should be used to cluster

        Returns
        -------
            self : LDA or cluster model
                A fitted model
        """
        if method == None:
            method = self.method

        if m_clustering == None:
            m_clustering = KMeans

        self.text_corpus = text_corpus
        if not self.dirichlet_dict:
            self.dirichlet_dict = corpora.Dictionary(text_corpus)
            self.bow_corpus = [
                self.dirichlet_dict.doc2bow(text) for text in text_corpus
            ]

        if method == "lda":
            if not self.lda_model:
                self.lda_model = LdaModel(
                    corpus=self.bow_corpus,
                    num_topics=self.num_topics,
                    id2word=self.dirichlet_dict,
                    chunksize=len(self.bow_corpus),
                    passes=20,  # increase to run model more iterations
                    alpha="auto",
                    random_state=None,
                )

        else:
            self.cluster_model = m_clustering(self.num_topics)
            self.vec[method] = self._vectorize(
                texts=texts, text_corpus=self.text_corpus, method=method
            )
            self.cluster_model.fit(self.vec[method])
