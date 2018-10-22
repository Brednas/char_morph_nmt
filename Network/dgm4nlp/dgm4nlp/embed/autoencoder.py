"""
:Authors: - Wilker Aziz
"""
import sys
import os
import tempfile
import logging
import numpy as np
from keras import layers
from keras.models import Model
from keras.utils.np_utils import to_categorical
from dgm4nlp.blocks import Generator
from dgm4nlp import td
from dgm4nlp.data import create_synthetic_data
from dgm4nlp.embed.experiment import ExperimentWrapper
from dgm4nlp.metrics import accuracy
from dgm4nlp.metrics import categorical_crossentropy


def make_auto_encoder(vocab_size,
                      emb_dim,
                      longest_sequence=None,
                      emb_dropout=0.,
                      hidden_layers=[],
                      optimizer='rmsprop',
                      name='AE'):
    """

    :param vocab_size: size of softmax layer
    :param emb_dim: dimensionality of projection/embedding layer
    :param longest_sequence: number of time steps in batch, use None to infer (defaults to None)
    :param emb_dropout: dropout level for embedding layer
    :param hidden_layers: hidden layers between embeddings and prediction
        specified as a list of pairs (dimensionality, nonlinearity)
    :param optimizer: string or keras.optimizers.Optimizer object (defaults to 'rmsprop')
    :param name: name of the computation graph (and prefix for its layers)
    :return: compiled Model(input=x, output=P(X|z))
    """
    # Input sequences
    # (B, M)
    x = layers.Input(shape=(longest_sequence,), dtype='int64', name='{}.X'.format(name))

    # 1. Inference network
    # Encoder's embedding layer
    embeddings = layers.Embedding(input_dim=vocab_size,
                                  output_dim=emb_dim,
                                  input_length=longest_sequence,
                                  mask_zero=False,
                                  name='{}.Embedding'.format(name))(x)

    # 2. Generative model P(X|Z=z)
    # (B, M, Vx)
    px = td.Softmax(vocab_size, hidden_layers=hidden_layers, name='{}.px'.format(name))(embeddings)

    model = Model(inputs=x, outputs=px, name='AE')
    model.compile(optimizer=optimizer,
                  loss=categorical_crossentropy,
                  metrics=[accuracy])

    return model


class AEGenerator(Generator):
    """
    Keras-compatible batch generator for the auto-encoder
    """

    def __init__(self, nb_classes,
                 shorter_batch='trim',
                 endless=True,
                 dynamic_sequence_length=False):
        super(AEGenerator, self).__init__(shorter_batch=shorter_batch,
                                          endless=endless,
                                          dynamic_sequence_length=dynamic_sequence_length)
        self.nb_classes = nb_classes

    def get(self, corpus, batch_size):
        for x, m in corpus.batch_iterator(batch_size,
                                          endless=self.endless,
                                          shorter_batch=self.shorter_batch,
                                          dynamic_sequence_length=self.dynamic_sequence_length):
            x_onehot = np.reshape(to_categorical(x.flatten(), self.nb_classes),
                           (x.shape[0], x.shape[1], self.nb_classes))
            yield x, x_onehot


def test_auto_encoder(training_path,
                      validation_path,
                      output_dir,
                      vocab_size=1000,
                      shortest=1,
                      longest=50,
                      dynamic_sequence_length=False,
                      # architecture parameters
                      emb_dim=128,
                      emb_dropout=0.,
                      hidden_layers=[],
                      # optimisation parameters
                      batch_size=1000,
                      nb_epochs=10,
                      optimizer='adagrad',
                      # convergence criteria
                      early_stopping=None):

    experiment = ExperimentWrapper(training_path=training_path,
                                   validation_path=validation_path,
                                   nb_words=vocab_size,
                                   shortest_sequence=shortest,
                                   longest_sequence=longest,
                                   output_dir=output_dir,
                                   dynamic_sequence_length=dynamic_sequence_length)

    # this gets the model
    model = make_auto_encoder(vocab_size=experiment.tk.vocab_size(),
                              emb_dim=emb_dim,
                              longest_sequence=None if dynamic_sequence_length else experiment.training.longest_sequence(),
                              emb_dropout=emb_dropout,
                              hidden_layers=hidden_layers,
                              optimizer=optimizer)

    # this makes a generator for this type of model
    generator = AEGenerator(experiment.tk.vocab_size(), dynamic_sequence_length=dynamic_sequence_length)

    # here we fit the model using batches from generator
    experiment.fit(model,
                   batch_size=batch_size,
                   nb_epochs=nb_epochs,
                   generator=generator,
                   early_stopping=early_stopping)

