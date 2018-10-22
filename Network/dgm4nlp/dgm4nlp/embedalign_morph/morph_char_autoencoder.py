"""
:Authors: - Wilker Aziz, adapted for characters and morphemes by Sander Bijl de Vroe
"""
import numpy as np
from dgm4nlp import ibm
from dgm4nlp import td
from dgm4nlp.blocks import Generator
from dgm4nlp.embedalign_morph.morph_experiment import ExperimentWrapper
from dgm4nlp.metrics import accuracy
from dgm4nlp.metrics import categorical_crossentropy
from dgm4nlp.nlpcharutils import CharTokenizer
from dgm4nlp.nlpmorphutils import MorphTokenizer
from dgm4nlp.nlputils import Tokenizer
from keras import backend as K
from keras import layers
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import TimeDistributed

# Reshaping functions for use in lambda layers

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

def remove_middle_dim4(t4):
    return K.reshape(t4, (K.shape(t4)[0],K.shape(t4)[1],K.shape(t4)[3]))

# Converts a t3 back to t4
# that is we reintroduce the word-step dimension
def back_to_t4(t3, x):  # (B * M, C, dc)
    return K.reshape(t3, (K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], K.shape(t3)[2]))  # (B, M, C, dc)
# Usage example:
# c = layers.Lambda(lambda t: back_to_t4(t), output_shape=(longest_xsnt, longest_xword, dc), name='c')(emb_c)

# Converts t2 back to t3, reintroducing word dimension
# From (B*M, dx) to (B, M, dx)
def back_to_t3(t2, x):
    return K.reshape(t2, (K.shape(x)[0], K.shape(x)[1], K.shape(t2)[1]))

def collapse_M_Mph(t4):
    return K.reshape(t4, (K.shape(t4)[0], K.shape(t4)[1]*K.shape(t4)[2], K.shape(t4)[3]))
    # return K.reshape(t4, (K.shape(t4)[0], -1, K.shape(t4)[3]))

def remove_last(t3):
    return t3[:,:,:-1]

def M(t4):
    return K.shape(t4[0,:,0,0])

def M_Mph(t4):
    pass

def dummy(t3):
    return t3



class BitextGenerator(Generator):
    """
    Keras generator for batches over bilingual corpora.
    """

    def __init__(self, nb_classes, nb_morphs,
                 shorter_batch='trim',
                 endless=True,
                 dynamic_sequence_length=False):
        super(BitextGenerator, self).__init__(shorter_batch=shorter_batch,
                                              endless=endless,
                                              dynamic_sequence_length=dynamic_sequence_length)
        self.nb_classes = nb_classes
        self.nb_morphs = nb_morphs

    def get(self, tritext, batch_size):
        for (x, mx), (y, my), (morph_data, masked_morphs) in tritext.batch_iterator(batch_size,
                                                      endless=self.endless,
                                                      shorter_batch=self.shorter_batch,
                                                      dynamic_sequence_length=self.dynamic_sequence_length):
            # B, N, Vy
            y_labels = np.reshape(to_categorical(y.flatten(), self.nb_classes),
                                  (y.shape[0], y.shape[1], self.nb_classes))
            # B, M* Mph, Vmph
            print(morph_data.shape[0], morph_data.shape[1] * morph_data.shape[2], self.nb_morphs)
            m_labels = np.reshape(to_categorical(morph_data.flatten(), self.nb_morphs),
                                   (morph_data.shape[0], morph_data.shape[1] * morph_data.shape[2], self.nb_morphs))
            #m_labels = np.reshape(to_categorical(morph_data.flatten(), self.nb_morphs),
            #                      (morph_data.shape[0], morph_data.shape[1], morph_data.shape[2], self.nb_morphs))
            print(m_labels.shape)
            yield [x, y, morph_data], {'AE.py_x' : y_labels, 'AE.pm_x': m_labels}


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

def make_morph_auto_encoder(x_vocab_size,
                            y_vocab_size,
                            x_char_vocab_size,
                           x_morph_vocab_size,
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
                            most_xmorphs=None,
                            longest_ysnt=None,
                           morph_emb_size = 64,
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
        rnn_units = char_emb_dim

    # Input sequences
    # (B, M, C)
    x = layers.Input(shape=(longest_xsnt,longest_xword), dtype='int64', name='{}.X'.format(name))
    # (B, N)
    y = layers.Input(shape=(longest_ysnt,), dtype='int64', name='{}.Y'.format(name))
    # (B, M, Mph)
    m = layers.Input(shape=(longest_xsnt,most_xmorphs), dtype='int64', name='{}.M'.format(name))

    # Embed words (B , M, C, dc)
    emb_c = TimeDistributed(layers.Embedding(input_dim=x_char_vocab_size, output_dim=char_emb_dim, mask_zero=False), name='embed_c')(x)

    # Apply convolutions over embeddings, arriving at one embedding for each word
    # From (B , M, C, dc) to (B, M, dx)
    # Rough/hardcoded version - defines convolution/maxpooling 3 times for each kernel size
    conv_c2 = TimeDistributed(layers.convolutional.Conv1D(filters=num_filters[0], kernel_size=2, strides=1, padding='valid',
                                          activation=cnn_activation), name='conv_c2')(emb_c)
    pool_c2 = TimeDistributed(layers.convolutional.MaxPooling1D(pool_size=longest_xword - (2 - 1)), name='pool_c2')(conv_c2)

    conv_c3 = TimeDistributed(layers.convolutional.Conv1D(filters=num_filters[1], kernel_size=3, strides=1, padding='valid',
                                          activation=cnn_activation), name='conv_c3')(emb_c)
    pool_c3 = TimeDistributed(layers.convolutional.MaxPooling1D(pool_size=longest_xword - (3 - 1)), name='pool_c3')(conv_c3)

    conv_c4 = TimeDistributed(layers.convolutional.Conv1D(filters=num_filters[2], kernel_size=4, strides=1, padding='valid',
                                          activation=cnn_activation), name='conv_c4')(emb_c)
    pool_c4 = TimeDistributed(layers.convolutional.MaxPooling1D(pool_size=longest_xword - (4 - 1)), name='pool_c4')(conv_c4)

    # Concatenate the output of the convolutions with different kernel_sizes
    # from 3 (B, M, 1, dx/3) tensors to (B, M, 1, dx); then (B, M, dx)
    conv_concat = Concatenate(axis=-1)([pool_c2, pool_c3, pool_c4])
    x_embeddings = layers.Lambda(lambda t: remove_middle_dim4(t), output_shape=(longest_xsnt,sum(num_filters)), name='dim2_c')(conv_concat)

    if nb_birnn_layers > 0:  # apply a number of BiRNN layers
        x_embeddings = td.BiRNN(rnn_units,
                              dropout=rnn_dropout,
                              recurrent_dropout=rnn_recurrent_dropout,
                              merge_mode=rnn_merge_mode,
                              nb_layers=nb_birnn_layers,
                              architecture=rnn_architecture,
                              name='{}.X-BiRNN'.format(name), output_shape = (longest_xsnt,rnn_units))(x_embeddings)


    # 2. Generative model P(M|X)
    # pm_x gets compared against the categorical morph label data.

    # (B,M,Mph-1) Ignores last morpheme since we predict nothing from it
    m_remove_last = layers.Lambda(lambda t: remove_last(t),
                                  output_shape=(longest_xsnt, most_xmorphs - 1),
                                  name = 'm_remove_last')(m)
    # (B,M,Mph-1,Mx) Embedding morphemes
    m_emb = TimeDistributed(layers.Embedding(input_dim = x_morph_vocab_size,
                                             output_dim = morph_emb_size,
                                             mask_zero = False),
                                             input_shape = (longest_xsnt,most_xmorphs-1),
                                             name = 'embed_m')(m_remove_last)
    # (B,M,Mph,mx) Adding an M0 to the beginning of each morpheme sequence
    m_with_zeros = TimeDistributed(layers.convolutional.ZeroPadding1D(padding=[1, 0]))(m_emb)

    # (B,M,Mph,dx) Repeating word xi
    xrep = TimeDistributed(layers.core.RepeatVector(most_xmorphs))(x_embeddings)
    # (B,M,Mph,dx+mx) Concatenate each morpheme to a copy of the word
    xm = Concatenate(axis=-1, name='{}.xm'.format(name))([xrep, m_with_zeros])

    # (B,M,Mph,Vm) Recurrent layer arriving at Vm
    xm_gru = TimeDistributed(layers.recurrent.GRU(units=x_morph_vocab_size,
                               unroll=True,
                               return_sequences=True,
                               implementation=2,
                               input_dim=rnn_units + morph_emb_size,
                               input_length=most_xmorphs + 1),
                               name='rec')(xm)

    # (B,M,Mph,Vm) Softmax over the RNN's output
    pm_x_t4 = layers.Activation('softmax')(xm_gru)
    pm_x = layers.Lambda(lambda t: collapse_M_Mph(t),name = '{}.pm_x'.format(name), output_shape=(None, x_morph_vocab_size))(pm_x_t4)

    # 3. Generative model P(Y|X=x)

    # 3.a Build an alignment model P(A|X,M,N)
    # (B, N, M)
    x_2D = layers.Lambda(lambda t: K.sum(x, axis=-1))(x)
    pa_x = alignment_component(longest_xsnt, longest_ysnt, name='{}.pa_x'.format(name))([x_2D, y])

    # 3.b P(Y|X,A) = P(Y|X_A)
    # (B, M, Vy)
    py_xa = td.Softmax(y_vocab_size, hidden_layers=hidden_layers, name='{}.py_xa'.format(name))(x_embeddings)

    # Deterministic ATTENTION
    # for every (i, j):
    #   sim = softmax(FFNN(x_i, y_{j-1}))  for every (i, j)
    #   cj = \sum_i sim_ij e_i
    #   this requires embedding y_{j-1}
    # then P(Y|x) = softmax(FFNN(c))

    # 3.c Marginalise alignments: \sum_a P(a|x) P(Y|x,a)
    # (B, N, Vy)
    py_x = layers.Lambda(lambda pair: K.batch_dot(pair[0], pair[1]),
                         output_shape=(longest_ysnt, y_vocab_size),
                         name='{}.py_x'.format(name))([pa_x, py_xa])

    model = Model(inputs=[x, y, m], outputs=[py_x, pm_x], name=name)
    model.compile(optimizer=optimizer,
                  loss={'{}.py_x'.format(name): categorical_crossentropy,
                        '{}.pm_x'.format(name): categorical_crossentropy},
                  metrics={'{}.py_x'.format(name): accuracy,
                            '{}.pm_x'.format(name): accuracy})

    return model


def test_morph_auto_encoder(training_paths,
                            validation_paths=[],
                            test_paths=[],
                            tok_classes = [CharTokenizer, Tokenizer, MorphTokenizer],
                            # data generation
                            vocab_size=[1000, 1000, 1000],
                            x_char_vocab_size = 500,
                            x_morph_vocab_size = 200,
                            shortest=[3, 3],
                            longest=[15, 15],
                            dynamic_sequence_length=False,
                            # architecture parameters
                            char_emb_dim=128,
                            num_filters = 10,
                            morph_emb_size = 64,
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

    assert len(training_paths) == 3, 'I expect tritexts'

    # get some random file names
    experiment = ExperimentWrapper(training_paths=training_paths,
                                   validation_paths=validation_paths,
                                   test_paths=test_paths,
                                   tok_classes = tok_classes,
                                   nb_words=vocab_size,
                                   shortest_sequence=shortest,
                                   longest_sequence=longest,
                                   output_dir=output_dir,
                                   bos_str=['-NULL-', None, '-NULL-'],  # for now conditional models always use a fake NULL input
                                   dynamic_sequence_length=dynamic_sequence_length)

    # this gets the model
    model = make_morph_auto_encoder(x_vocab_size=experiment.tks[0].vocab_size(),
                                    y_vocab_size=experiment.tks[1].vocab_size(),
                                    x_char_vocab_size = experiment.tks[1].vocab_size(),
                                    x_morph_vocab_size = experiment.tks[2].vocab_size(),
                                    # encoder
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
                                    most_xmorphs = experiment.tks[2].longest_xword(),
                                    longest_ysnt=None if dynamic_sequence_length else experiment.training.longest_sequence(1),
                                    morph_emb_size = morph_emb_size,
                                    # optimisation
                                    optimizer=optimizer)



    # this makes a generator for this type of model
    generator = BitextGenerator(experiment.tks[1].vocab_size(), # y's vocab
                                experiment.tks[2].vocab_size(), # morpheme vocab
                                dynamic_sequence_length=dynamic_sequence_length)

    # here we fit the model using batches from generator
    experiment.run(model,
                   batch_size=batch_size,
                   nb_epochs=nb_epochs,
                   generator=generator,
                   viterbifunc=viterbi_alignments,
                   early_stopping=early_stopping)
