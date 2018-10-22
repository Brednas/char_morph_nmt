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
from dgm4nlp.blocks import Generator
from dgm4nlp.metrics import accuracy
from dgm4nlp.metrics import categorical_crossentropy
from dgm4nlp.callback import EarlyStoppingChain
from dgm4nlp.embedalign.experiment import ExperimentWrapper


class BitextGenerator(Generator):
    """
    Keras generator for batches over bilingual corpora.
    """

    def __init__(self, longest_y, nb_classes):
        self.longest_y = longest_y
        self.nb_classes = nb_classes

    def get(self, bitext, batch_size, shorter_batch='trim', endless=True):
        for (x, mx), (y, my) in bitext.batch_iterator(batch_size, endless=endless, shorter_batch=shorter_batch):
            y_labels = np.reshape(to_categorical(y.flatten(), self.nb_classes),
                                  (y.shape[0], self.longest_y, self.nb_classes))
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


def make_auto_encoder(longest_x, longest_y,
                      x_vocab_size, y_vocab_size,
                      dx,
                      dy,
                      emb_dropout=0.,
                      selection_layers=[],
                      insertion_layers=[],
                      translation_layers=[],
                      hidden_layers=[],
                      # alignment component
                      alignment_component: ibm.AlignmentComponent = ibm.UniformAlignmentComponent(),
                      # optimiser parameters
                      optimizer='rmsprop',
                      name='AE'):
    """
    P(Y_1^n|X_0^m) = \prod_{j=1}^n \sum_{i=0}^m P(A_j=i|m, n) P(Y_j|X_i)

    Important note: 0 is considered a mask value

    :param longest_x: longest input sequence (M)
    :param longest_y: longest output sequence (N)
    :param x_vocab_size: size of input vocabulary (Vx)
    :param y_vocab_size: size of output vocabulary (Vy)
    :param emb_dim: dimensionality of projection/embedding layer (dx)
    :param emb_dropout: dropout level for embedding layer
    :param hidden_layers: hidden layers between embedding and precition
        specified as a list of pairs (dimensionality, nonlinearity)
    :param alignment_component: an instance of AlignmentComponent
    :param optimizer: string or keras.optimizers.Optimizer object (defaults to 'rmsprop')
    :param name: name of the computation graph (and prefix for its layers)
    :return: compiled Model(input=x, output=P(X|z))
    """
    # Input sequences
    # (B, M)
    x = layers.Input(shape=(longest_x,), dtype='int64', name='{}.X'.format(name))
    # (B, N)
    y = layers.Input(shape=(longest_y,), dtype='int64', name='{}.Y'.format(name))

    # Source embeddings
    # (B, M, dx)
    x_emb = layers.Embedding(input_dim=x_vocab_size,
                                  output_dim=dx,
                                  input_length=longest_x,
                                  mask_zero=False,
                                  name='{}.X-Embedding'.format(name))(x)

    # Target embeddings
    # (B, N, dy)
    y_emb = layers.Embedding(input_dim=y_vocab_size,
                             output_dim=dy,
                             input_length=longest_y,
                             mask_zero=False,
                             name='{}.Y-Embedding'.format(name))(y)

    # 2. Generative model P(Y|X=x)

    # 2.a Build a distribution P(S|Y_prev) to select between insertion and translation
    #  here I am assuming y[:, 0] to be padding
    # (B, N, 2)
    ps = td.Softmax(2, hidden_layers=selection_layers, name='{}.ps'.format(name))(y_emb)

    # 2.b Build a language model P(Y|Y_prev) used to insert words
    # (B, N, Vy)
    py_yprev = td.Softmax(y_vocab_size, hidden_layers=insertion_layers, name='{}.py_yprev'.format(name))(y_emb)

    # 2.c Build a lexical model P(Y|X) used to translate words
    # (B, M, Vy)
    py_xa = td.Softmax(y_vocab_size, hidden_layers=translation_layers, name='{}.py_xa'.format(name))(x_emb)

    # 2.d Build an alignment model P(A|m,n)
    # (B, N, M)
    pa_x = alignment_component(longest_x, longest_y, name='{}.pa_x'.format(name))([x, y])

    # 2.e Marginalise alignments: \sum_a P(a|x) P(Y|x,a)
    # (B, N, Vy)
    py_x = layers.Lambda(lambda pair: K.batch_dot(pair[0], pair[1]),
                         output_shape=(longest_y, y_vocab_size),
                         name='{}.py_x'.format(name))([pa_x, py_xa])

    # 2.f Marginalise selection variables: P(S=1|Y_prev) P(Y|Y_prev) + P(S=0|Y_prev)P(Y|x_1^m)
    # (B, N, Vy)
    layers.Lambda(lambda t: t[0][:, :, 0][:, :, None] * t[1] + t[0][:, :, 1][:, :, None] * t[2])([ps, py_yprev, py_x])



    model = Model(inputs=[x, y], outputs=[py_x, pa_x, py_xa], name=name)
    model.compile(optimizer=optimizer,
                  loss={'{}.py_x'.format(name): categorical_crossentropy,
                        '{}.pa_x'.format(name): None,
                        '{}.py_xa'.format(name): None},
                  metrics={'{}.py_x'.format(name): accuracy})

    return model


def test_auto_encoder(training_paths,
                      validation_paths,
                      # data generation
                      vocab_size=[1000, 1000],
                      shortest=[3, 3],
                      longest=[15, 15],
                      # architecture parameters
                      emb_dim=128,
                      emb_dropout=0.,
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
                                   nb_words=vocab_size,
                                   shortest_sequence=shortest,
                                   longest_sequence=longest,
                                   output_dir=output_dir,
                                   bos_str=['-NULL-', None])  # for now conditional models always use a fake NULL input

    # this gets the model
    model = make_auto_encoder(longest_x=experiment.training.longest_sequence(0),
                              longest_y=experiment.training.longest_sequence(1),
                              x_vocab_size=experiment.tks[0].vocab_size(),
                              y_vocab_size=experiment.tks[1].vocab_size(),
                              # encoder
                              emb_dim=emb_dim,
                              emb_dropout=emb_dropout,
                              # lexical distribution
                              hidden_layers=hidden_layers,
                              # alignment distribution
                              alignment_component=alignment_component,
                              # optimisation
                              optimizer=optimizer)

    # this makes a generator for this type of model
    generator = BitextGenerator(experiment.training.longest_sequence(1), experiment.tks[1].vocab_size())

    # here we fit the model using batches from generator
    experiment.run(model,
                   batch_size=batch_size,
                   nb_epochs=nb_epochs,
                   generator=generator,
                   viterbifunc=viterbi_alignments,
                   early_stopping=early_stopping)


if __name__ == '__main__':
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    # import theano
    # theano.config.profile = True
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    stopping_criteria = EarlyStoppingChain(monitor=['val_loss', 'val_aer'],
                                           min_delta=[0.1, 0.01],
                                           patience=[10, 5],
                                           mode=['min', 'min'],
                                           initial_epoch=5,
                                           verbose=True)

    uni_cpd = ibm.UniformAlignmentComponent()
    # layers: [(30, 'relu')]
    #pos_cpd = ibm.PositionCPDComponent(dm=30, dn=30, dj=30, context_layers=[], dynamic_support=False)

    test_auto_encoder(training_paths=['/home/wferrei1/github/dgm4nlp/data/en-fr/training.en-fr.en',
                                      '/home/wferrei1/github/dgm4nlp/data/en-fr/training.en-fr.fr'],
                      validation_paths=['/home/wferrei1/github/dgm4nlp/data/en-fr/trial.en-fr.en',
                                        '/home/wferrei1/github/dgm4nlp/data/en-fr/trial.en-fr.fr',
                                        '/home/wferrei1/github/dgm4nlp/data/en-fr/trial.en-fr.naacl'],
                      # data generation
                      vocab_size=[1000, 1000],
                      shortest=[3, 3],
                      longest=[50, 50],
                      # architecture parameters
                      emb_dim=256,
                      emb_dropout=0.,
                      hidden_layers=[(256, 'tanh')],
                      # alignment distribution options
                      alignment_component=uni_cpd,
                      # optimisation parameters
                      batch_size=100,
                      nb_epochs=100,
                      optimizer=optimizers.Adam(),
                      # convergence criteria
                      early_stopping=stopping_criteria,
                      # output files
                      output_dir='/home/wferrei1/github/dgm4nlp/debug')


