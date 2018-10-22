"""
:Authors: - Wilker Aziz, adapted for characters by Sander Bijl de Vroe
"""
import numpy as np
from dgm4nlp import ibm
from dgm4nlp import td
from dgm4nlp.blocks import Generator
from dgm4nlp.embedalign_char.char_experiment import ExperimentWrapper
from dgm4nlp.metrics import accuracy
from dgm4nlp.metrics import categorical_crossentropy
from keras import backend as K
from keras import layers
from keras.layers import Activation
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils.np_utils import to_categorical


# Flattens a t3 tensor to t2, keeping the final (character) dimension intact
def collapse_BandM(t):  # (B, M, C)
    # B, M, C = K.shape(t)[0], K.shape(t)[1], K.shape(t)[2]
    shape = K.shape(t)
    flat_c = K.reshape(t, (-1, K.shape(t)[-1]))  # (B * M, C)
    return flat_c

# Removes middle dimension left from max pooling
# from (B*M, 1, dx) to (B*M, dx)
def remove_middle_dim(t1_3):
    return K.reshape(t1_3, (K.shape(t1_3)[0], K.shape(t1_3)[2]))

# Converts a t3 back to t4
# that is we reintroduce the word-step dimension
def back_to_t4(t3, x, char_emb_dim):  # (B * M, C, dc)
    return K.reshape(t3, (-1, K.shape(x)[1], K.shape(x)[2], char_emb_dim))  # (B, M, C, dc)
# Usage example:
# c = layers.Lambda(lambda t: back_to_t4(t), output_shape=(longest_xsnt, longest_xword, dc), name='c')(emb_c)

# Converts t2 back to t3, reintroducing word dimension
# From (B*M, dx) to (B, M, dx)
def back_to_t3(t2, x):
    return K.reshape(t2, (K.shape(x)[0], K.shape(x)[1], K.shape(t2)[1]))



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

    # mx / my are masks.
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

def make_char_auto_encoder(x_vocab_size,
                            y_vocab_size,
                            x_char_vocab_size,
                            char_emb_dim = 10,
                            num_filters = [10,10,10],
                            emb_dropout=0.,
                           cnn_activation = 'relu',
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
                            longest_xsnt=None,
                           longest_xword=None,
                            longest_ysnt=None,
                            # optimiser parameters
                            optimizer='rmsprop',
                            name='AE'):
    #vocab_chars_x, dc = 10, num_filters = 10, longest_xsnt = None, longest_xword = None

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
        rnn_units = char_emb_dim

    # Input sequences
    # (B, M, C)
    x = layers.Input(shape=(longest_xsnt,longest_xword), dtype='int64', name='{}.X'.format(name))
    # (B, N)
    y = layers.Input(shape=(longest_ysnt,), dtype='int64', name='{}.Y'.format(name))

    # 1. Inference network
    # Encoder's embedding layer - Convolutions over characters, eventually arriving at vectors per word
    # (B, M, dx)

    # here we get rid of the time dimension (word steps)
    # From (B, M, C) to (B*M, C)
    flat_c = layers.Lambda(lambda t: collapse_BandM(t), output_shape=(longest_xword,), name='flat_c')(x)

    # now we can embed words easily (it takes embedding char steps) (B * M, C, dc)
    emb_c = layers.Embedding(input_dim=x_char_vocab_size, output_dim=char_emb_dim, mask_zero=False, name='embed_c')(flat_c)

    # Apply convolutions over embeddings, arriving at one embedding for each word
    # From (B*M, C, dc) to (B*M, dx)
    # Rough/hardcoded version - defines convolution/maxpooling 3 times for each kernel size
    conv_c2 = layers.convolutional.Conv1D(filters=num_filters[0], kernel_size=2, strides=1, padding='valid',
                                          activation=cnn_activation, name='conv_c2')(emb_c)
    pool_c2 = layers.convolutional.MaxPooling1D(pool_size=longest_xword - (2 - 1), name='pool_c2')(conv_c2)

    conv_c3 = layers.convolutional.Conv1D(filters=num_filters[1], kernel_size=3, strides=1, padding='valid',
                                          activation=cnn_activation, name='conv_c3')(emb_c)
    pool_c3 = layers.convolutional.MaxPooling1D(pool_size=longest_xword - (3 - 1), name='pool_c3')(conv_c3)

    conv_c4 = layers.convolutional.Conv1D(filters=num_filters[2], kernel_size=4, strides=1, padding='valid',
                                          activation=cnn_activation, name='conv_c4')(emb_c)
    pool_c4 = layers.convolutional.MaxPooling1D(pool_size=longest_xword - (4 - 1), name='pool_c4')(conv_c4)

    # Concatenate the output of the convolutions with different kernel_sizes
    # from 3 (B*M, 1, dx/3) tensors to (B*M, 1, dx)
    conv_concat = Concatenate(axis=-1)([pool_c2, pool_c3, pool_c4])

    dim2_c = layers.Lambda(lambda t: remove_middle_dim(t), output_shape=(sum(num_filters),), name='dim2_c')(conv_concat)

    x_embeddings = layers.Lambda(lambda pair: back_to_t3(pair[0], pair[1]), output_shape=(longest_xsnt, sum(num_filters)), name='word_emb_x')(
        [dim2_c, x])


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
    x_2D = layers.Lambda(lambda t: K.sum(x, axis=-1))(x)
    pa_x = alignment_component(longest_xsnt, longest_ysnt, name='{}.pa_x'.format(name))([x_2D, y])

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
                         output_shape=(longest_ysnt, y_vocab_size),
                         name='{}.py_x'.format(name))([pa_x, py_xa])

    model = Model(inputs=[x, y], outputs=[py_x, pa_x, py_xa], name=name)
    model.compile(optimizer=optimizer,
                  loss={'{}.py_x'.format(name): categorical_crossentropy,
                        '{}.pa_x'.format(name): None,
                        '{}.py_xa'.format(name): None},
                  metrics={'{}.py_x'.format(name): accuracy})

    return model


def test_char_auto_encoder(training_paths,
                      validation_paths=[],
                      test_paths=[],
                      # data generation
                      vocab_size=[1000, 1000],
                      x_char_vocab_size = 50,
                      shortest=[3, 3],
                      longest=[15, 15],
                      dynamic_sequence_length=False,
                      # architecture parameters
                      char_emb_dim=128,
                      cnn_activation='tanh',
                      num_filters = [10,10,10],
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
    model = make_char_auto_encoder(x_vocab_size=experiment.tks[0].vocab_size(),
                                    y_vocab_size=experiment.tks[1].vocab_size(),
                                    x_char_vocab_size = x_char_vocab_size,
                                    # encoder
                                    cnn_activation=cnn_activation,
                                    char_emb_dim=char_emb_dim,
                                    num_filters=num_filters,
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
                                    longest_xsnt=None if dynamic_sequence_length else experiment.training.longest_sequence(0),
                                    longest_xword=experiment.tks[0].longest_xword(),
                                    longest_ysnt=None if dynamic_sequence_length else experiment.training.longest_sequence(1),
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
