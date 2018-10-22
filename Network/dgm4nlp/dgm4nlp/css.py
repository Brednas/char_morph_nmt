"""
:Authors: - Wilker Aziz
"""
import numpy as np
from keras import backend as K
from keras import layers
from keras.models import Model
from dgm4nlp.metrics import categorical_crossentropy


def dot(inputs):
    # (B, M, dx), (B, M, S, dc=dx)
    a, b = inputs
    return K.sum(a[:, :, None, :] * b[:, None, :, :], axis=-1)


def make_css_net(vocab_size, dx, nb_classes=None, dc=None, longest=1, support_size=None):
    if nb_classes is None:
        nb_classes = vocab_size
    if dc is None:
        dc = dx
    elif dc != dx:
        pass  # TODO: use a projection matrix
    # (B, M)  inputs
    x = layers.Input(shape=(longest,), dtype='int64')
    # (B, M, dx)
    x_emb = layers.Embedding(input_dim=vocab_size, output_dim=dx, input_length=longest)(x)

    # (B, M, S)  classes
    c = layers.Input(shape=(longest, support_size), dtype='int64')
    k = layers.Input(shape=(longest, support_size), dtype='float32')
    # (B, M, S, dc)
    c_emb = layers.Embedding(input_dim=nb_classes, output_dim=dc)(c)

    # (B, M, S)  score the truncated support
    log_u = layers.Lambda(dot, output_shape=(longest, longest))([x_emb, c_emb])
    log_k = layers.Lambda(K.log, output_shape=(longest, support_size))(k)
    log_u = layers.Add([log_u, log_k])
    p = layers.Activation('softmax')(log_u)

    model = Model(inputs=[x, c, k], outputs=p)
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy)


def truncated_support(batch, nb_classes, nb_samples):
    # batch is (B, M)
    # (B, M, S)
    samples = np.zeros((batch.shape[0], batch.shape[1], nb_samples), dtype='int64')
    targets = np.zeros((batch.shape[0], batch.shape[1], nb_samples), dtype='int64')
    for i, sequences in enumerate(batch):
        for j, x in enumerate(x):
            # distribution over classes
            p = np.ones(nb_classes)
            # no chance to sample positive classes
            p[x] = 0.0
            # make the distribution uniform over negative classes (TODO: use class frequency)
            p /= p.sum()
            # select without replacement a subset
            samples[i, j] = np.random.choice(nb_classes, size=nb_samples, replace=False, p=p)
            samples[i, j, 0] = x
            targets[i, j, 0] = 1
    return samples, targets


def main():
    n_samples = 100
    vocab_size = 100
    longest = 1
    support_size = 20
    # positive
    X = np.random.randint(1, vocab_size, size=(n_samples, longest))
    # negative
    S, Y = truncated_support(X, n_samples, support_size)
    # truncated support (includes correct classes)
    S = np.concatenate((X, N), axis=1)
    # TODO: get the correct Bernoulli parameter
    b = (support_size - 1) / (vocab_size - 1)
    # TODO: get the correct Kappa matrix
    K = np.full((n_samples, support_size), b)
    K[:, 0] = 1.0
    # This indicates where the correct class is in S
    Y = np.zeros((n_samples, longest, support_size), dtype='int64')
    Y[:, :, 0] = 1
    #outputs = model.predict([X, S, K])
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)

    #print(X)
    #print(S)
    #model.fit([X, S, K], Y)

