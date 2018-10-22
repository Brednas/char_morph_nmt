"""
:Authors: - Wilker Aziz
"""
import logging
import sys
import os
import numpy as np
from keras import layers
from keras import backend as K
from keras.models import Model
from keras import optimizers
from keras.utils.np_utils import to_categorical
from dgm4nlp import td
from dgm4nlp import ibm
from dgm4nlp import blocks
from dgm4nlp.blocks import Generator
from dgm4nlp.metrics import accuracy
from dgm4nlp.metrics import categorical_crossentropy
from dgm4nlp.callback import EarlyStoppingChain
from dgm4nlp.embedalign.experiment import ExperimentWrapper
from theano.tensor.nnet import h_softmax



class BitextGenerator(Generator):
    """
    Keras generator for batches over bilingual corpora.
    """

    def __init__(self, nb_classes,
                 shorter_batch='trim',
                 endless=True,
                 dynamic_sequence_length=False):
        super(BitextGenerator, self).__init__(shorter_batch=shorter_batch,
                                              endless=endless,
                                              dynamic_sequence_length=dynamic_sequence_length)
        self.nb_classes = nb_classes

    def get(self, bitext, batch_size):
        for (x, mx), (y, my) in bitext.batch_iterator(batch_size,
                                                      endless=self.endless,
                                                      shorter_batch=self.shorter_batch,
                                                      dynamic_sequence_length=self.dynamic_sequence_length):
            y_labels = np.reshape(to_categorical(y.flatten(), self.nb_classes),
                                  (y.shape[0], y.shape[1], self.nb_classes))
            yield [x, y], y_labels


def viterbi_alignments(inputs, predictions):
    """
    Viterbi alignments.

    :param inputs: (x,y) batch where x is (B, M) and y is (B, N)
    :param predictions: (py_x, pa_x, py_xa) batch where py_x is (B, N, Vy), pa_x is (B, N, M) and py_xa is (B, M, Vy)
    :return: alignments (B, N) and posteriors (B, N)
        where alignments[b, j] is the position in x[b] that generates y[b, j]
    """
    batch_x, batch_y = inputs
    batch_py_x, batch_pa_x, batch_py_xa = predictions
    B, N = batch_y.shape
    A = np.zeros((B, N), dtype='int64')
    P = np.zeros((B, N), dtype=K.floatx())
    for s, (x, y, pa_x, py_xa, py_x) in enumerate(zip(batch_x, batch_y, batch_pa_x, batch_py_xa, batch_py_x)):
        for j, yj in enumerate(y):
            if yj == 0:  # skip masked values
                break
            # (M,)
            paj_x = pa_x[j]
            # (M,)
            pyj_xa = py_xa[:, yj]
            # (M,)
            joint = paj_x * pyj_xa
            i = joint.argmax()
            p = joint[i] / py_x[j, yj]
            A[s, j] = i
            P[s, j] = p
            #print('Line %d: %d-%d (alignment only i=%d lexical only i=%d)' % (s + 1, i, j + 1, paj_x.argmax(), pyj_xa.argmax()))
            #print('pA', pa_x[j], pa_x[j].sum(-1))
            #print('pY|XA', pyj_xa, )
            #print('pYA|X', joint)
            #print('PA|XY', joint/py_x[j, yj])
            #print()
    return A, P


def make_auto_encoder(x_vocab_size, y_vocab_size,
                      emb_dim,
                      emb_dropout=0.,
                      nb_birnn_layers=0,
                      rnn_units=None,
                      rnn_dropout=0,
                      rnn_recurrent_dropout=0,
                      rnn_merge_mode='sum',
                      rnn_architecture='lstm',
                      hidden_layers=[],
                      # alignment component
                      alignment_component: ibm.AlignmentComponent = ibm.UniformAlignmentComponent(),
                      # shape
                      longest_x=None,
                      longest_y=None,
                      # optimiser parameters
                      optimizer='rmsprop',
                      name='AE'):
    """
    P(Y_1^n|X_0^m) = \prod_{j=1}^n \sum_{i=0}^m P(A_j=i|m, n) P(Y_j|X_i)

    Important note: 0 is considered a mask value

    :param x_vocab_size: size of input vocabulary (Vx)
    :param y_vocab_size: size of output vocabulary (Vy)
    :param emb_dim: dimensionality of projection/embedding layer (dx)
    :param emb_dropout: dropout after embedding layer
    :param nb_birnn_layers: number of BiLSTM layers applied on top of embedding matrix (defaults to 0)
    :param rnn_units: number of units per LSTM (defaults to emb_dim)
    :param rnn_dropout: LSTM dropout (defaults to 0)
    :param rnn_recurrent_dropout: LSTM recurrent dropout (defaults to 0)
    :param rnn_merge_mode: how to merge LSTM states (defaults to 'sum')
    :param rnn_architecture: one of 'lstm' or 'gru' (defaults to 'lstm')
    :param hidden_layers: hidden layers between embedding and precition
        specified as a list of pairs (dimensionality, nonlinearity)
    :param longest_x: longest input sequence (M), use None to infer (defaults to None)
    :param longest_y: longest output sequence (N), use None to infer (defaults to None)
    :param alignment_component: an instance of AlignmentComponent
    :param optimizer: string or keras.optimizers.Optimizer object (defaults to 'rmsprop')
    :param name: name of the computation graph (and prefix for its layers)
    :return: compiled Model(input=x, output=P(X|z))
    """

    if rnn_units is None:  # if not specified defaults to the embedding dimensionality
        rnn_units = emb_dim

    # Input sequences
    # (B, M)
    x = layers.Input(shape=(longest_x,), dtype='int64', name='{}.X'.format(name))
    # (B, N)
    y = layers.Input(shape=(longest_y,), dtype='int64', name='{}.Y'.format(name))

    # 1. Inference network
    # Encoder's embedding layer
    # (B, M, dx)
    x_embeddings = layers.Embedding(input_dim=x_vocab_size,
                                    output_dim=emb_dim,
                                    input_length=longest_x,
                                    mask_zero=True,
                                    name='{}.X-Embedding'.format(name))(x)

    if nb_birnn_layers > 0:  # apply a number of BiRNN layers
        x_embeddings = td.BiRNN(rnn_units,
                              dropout=rnn_dropout,
                              recurrent_dropout=rnn_recurrent_dropout,
                              merge_mode=rnn_merge_mode,
                              nb_layers=nb_birnn_layers,
                              architecture=rnn_architecture,
                              name='{}.X-BiRNN'.format(name))(x_embeddings)

    # 2. Generative model P(Y|X=x)

    # 2.a Build an alignment model P(A|X,M,N)
    # (B, N, M)
    pa_x = alignment_component(longest_x, longest_y, name='{}.pa_x'.format(name))([x, y])

    # 2.b P(Y|X,A) = P(Y|X_A)
    # (B, M, Vy)
    py_xa = td.Softmax(y_vocab_size, hidden_layers=hidden_layers, name='{}.py_xa'.format(name))(x_embeddings)

    # Deterministic ATTENTION
    # for every (i, j):
    #   sim = softmax(FFNN(x_i, y_{j-1}))  for every (i, j)
    #   cj = \sum_i sim_ij e_i
    #   this requires embedding y_{j-1}
    # then P(Y|x) = softmax(FFNN(c))

    # 2.c Marginalise alignments: \sum_a P(a|x) P(Y|x,a)
    # (B, N, Vy)
    py_x = layers.Lambda(lambda pair: K.batch_dot(pair[0], pair[1]),
                         output_shape=(longest_y, y_vocab_size),
                         name='{}.py_x'.format(name))([pa_x, py_xa])

    model = Model(inputs=[x, y], outputs=[py_x, pa_x, py_xa], name=name)
    model.compile(optimizer=optimizer,
                  loss={'{}.py_x'.format(name): categorical_crossentropy,
                        '{}.pa_x'.format(name): None,
                        '{}.py_xa'.format(name): None},
                  metrics={'{}.py_x'.format(name): accuracy})

    return model


def test_auto_encoder(training_paths,
                      validation_paths,
                      test_paths=[],
                      # data generation
                      vocab_size=[1000, 1000],
                      shortest=[3, 3],
                      longest=[15, 15],
                      dynamic_sequence_length=False,
                      # architecture parameters
                      emb_dim=128,
                      emb_dropout=0.,
                      nb_birnn_layers=0,
                      rnn_units=None,
                      rnn_dropout=0,
                      rnn_recurrent_dropout=0,
                      rnn_merge_mode='sum',
                      rnn_architecture='gru',
                      hidden_layers=[],
                      # alignment distribution options
                      alignment_component: ibm.AlignmentComponent=ibm.UniformAlignmentComponent(),
                      # optimisation parameters
                      batch_size=1000,
                      nb_epochs=1,
                      optimizer='adagrad',
                      # convergence criteria
                      early_stopping=None,
                      # output files
                      output_dir=None):

    assert len(training_paths) == 2, 'I expect bitexts'

    # get some random file names
    experiment = ExperimentWrapper(training_paths=training_paths,
                                   validation_paths=validation_paths,
                                   test_paths=test_paths,
                                   nb_words=vocab_size,
                                   shortest_sequence=shortest,
                                   longest_sequence=longest,
                                   output_dir=output_dir,
                                   bos_str=['-NULL-', None],  # for now conditional models always use a fake NULL input
                                   dynamic_sequence_length=dynamic_sequence_length)

    # this gets the model
    model = make_auto_encoder(x_vocab_size=experiment.tks[0].vocab_size(),
                              y_vocab_size=experiment.tks[1].vocab_size(),
                              # encoder
                              emb_dim=emb_dim,
                              emb_dropout=emb_dropout,
                              nb_birnn_layers=nb_birnn_layers,
                              rnn_units=rnn_units,
                              rnn_dropout=rnn_dropout,
                              rnn_recurrent_dropout=rnn_recurrent_dropout,
                              rnn_merge_mode=rnn_merge_mode,
                              rnn_architecture=rnn_architecture,
                              # lexical distribution
                              hidden_layers=hidden_layers,
                              # alignment distribution
                              alignment_component=alignment_component,
                              # shape
                              longest_x=None if dynamic_sequence_length else experiment.training.longest_sequence(0),
                              longest_y=None if dynamic_sequence_length else experiment.training.longest_sequence(1),
                              # optimisation
                              optimizer=optimizer)

    # this makes a generator for this type of model
    generator = BitextGenerator(experiment.tks[1].vocab_size(), dynamic_sequence_length=dynamic_sequence_length)

    # here we fit the model using batches from generator
    experiment.run(model,
                   batch_size=batch_size,
                   nb_epochs=nb_epochs,
                   generator=generator,
                   viterbifunc=viterbi_alignments,
                   early_stopping=early_stopping)
