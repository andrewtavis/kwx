"""
autoencoder
-----------

The autoencoder for combining topic modeling techniques

Contents:
    Autoencoder Class:
        _compile,
        fit
"""

import logging
import os
import warnings

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.model_selection import train_test_split

import keras
from keras.layers import Input, Dense
from keras.models import Model


class Autoencoder:
    """
    Autoencoder for learning latent space representation architecture (simplified for only one hidden layer)

    Notes
    -----
        Used to combine LDA and BERT vectors
    """

    def __init__(self, latent_dim=32, activation="relu", epochs=200, batch_size=128):
        """
        Parameters
        ----------
            latent_dim : int (default=32)
                The demensions of the dense layers

            activation : str (default=relu)
                The activation method for the keras model

            epochs : int (default=200)
                The number of modeling iterations

            batch_size : int (default=128)
                The number of data entries to feed into the modeling process in each epoch
        """
        # increase epochs to run model more iterations
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        Compile the computational graph

        Parameters
        ----------
            input_dim : int
                The dimension of the input

        Returns
        -------
            self.autoencoder : keras.models.Model
                A compiled keras model
        """
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(units=self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(units=input_dim, activation=self.activation)(encoded)

        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)

        encoded_input = Input(shape=(self.latent_dim,))
        decoder_layer = self.autoencoder.layers[-1]

        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        self.autoencoder.compile(optimizer="adam", loss=keras.losses.mean_squared_error)

    def fit(self, X):
        """
        Fit the model

        Parameters
        ----------
            X : np.array or list of lists
                The text data to be modeled

        Returns
        -------
            self.his : keras.models.Model
                A keras model fitted with the given parameters
        """
        if not self.autoencoder:
            self._compile(X.shape[1])

        X_train, X_test = train_test_split(X)
        self.his = self.autoencoder.fit(
            X_train,
            X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(X_test, X_test),
            verbose=0,
        )
