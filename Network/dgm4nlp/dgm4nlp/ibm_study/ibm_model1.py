"""
:Authors: - Adapted by Sander Bijl de Vroe, originally by Wilker Aziz
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
from dgm4nlp.embedalign.autoencoder import BitextGenerator
import theano

# P(aj_m)

# P(yj_xaj) = determined from embedding, MLP, softmax over vocabulary Vy

# batch dot implements marginalization

# P(aj_m)

# P(yj_xaj)
# sentence (or batch of sentences) comes as input

# sentence is embedded

# embeddings go through MLP; instantiated through Keras layers.

# softmax over the vocabulary;
def create_pa(xmask,ymask):
	return layers.Lambda(lambda masks: masks[0][:,None,:]+K.zeros_like(masks[1])[:,:,None])([xmask,ymask])

def normalize_t(x):
	return layers.Lambda(lambda t: x / K.sum(x,axis=1)[:, None])(x)

def make_mask(x):
	return layers.Lambda(lambda t: K.cast(K.not_equal(x, 0), K.floatx()))(x)
# sentences

# instantiates graph
def make_ibm1(vocab_size_tr,
              vocab_size_en,
              embedding_size,
              longest_tr=None,
              longest_en=None,
              optimizer='rmsprop',
              name='AE'):

    # BxM, where B=size of a batch; M=longest Turkish sentence
    x = layers.Input(shape=(longest_tr,), dtype='int64', name='{}.x'.format(name))
    # BxN, where N=longest English sentence
    y = layers.Input(shape=(longest_en,), dtype='int64', name='{}.y'.format(name))

    # 1. Create embeddings
    # produces BxMxD, where D = embedding dimension
    x_embedded = layers.Embedding(input_dim=vocab_size_tr,
                                    output_dim=embedding_size,
                                    input_length=longest_tr)(x)

    # 2. Embeddings go through MLP, with softmax over English vocabulary
    # produces BxMxVe
    py_xa = td.Softmax(vocab_size_en, hidden_layers=[(256, 'relu')], name='{}.py_xa'.format(name))(x_embedded)

    # 3. Calculate alignments and marginalize
    # Mask of Turkish input: BxM
    x_mask = make_mask(x)
    # Mask of English input: BxN
    y_mask = make_mask(y)
    # Turkish as uniform distribution determined by sentence length: BxM
    x_norm = normalize_t(x_mask)
    # Alignment probabilities (x' uniform distribution broadcasted into N): BxNxM
    pa = create_pa(x_norm, y_mask)
    # Marginalization over a to get py_x: BxNxVy
    # Produces tensor with probability distribution over English vocabulary for each
    # position in the English sentence (for each English sentence)
    py_x = layers.Lambda(lambda p_dists: K.batch_dot(p_dists[1], p_dists[0]), output_shape=
    (longest_en, vocab_size_en), name='{}.py_x'.format(name))([py_xa, pa])


    # Create model
    model = Model(inputs = [x,y], outputs = [py_x], name=name)

    # Compile model
    model.compile(optimizer=optimizer,
                  loss={'{}.py_x'.format(name): categorical_crossentropy},
                  metrics={'{}.py_x'.format(name): accuracy})

    return model


def train_ibm1(training_paths,
               validation_paths,
               vocab_size,
               shortest,
               longest,
               output_dir,
               dynamic_sequence_length,
               embedding_size,
               batch_size,
               nb_epochs,
               optimizer,
               early_stopping):

    experiment = ExperimentWrapper(training_paths=training_paths,
                                   validation_paths=validation_paths,
                                   nb_words=vocab_size,
                                   shortest_sequence=shortest,
                                   longest_sequence=longest,
                                   output_dir=output_dir,
                                   dynamic_sequence_length=dynamic_sequence_length)
                        # TODO bos; test_paths now missing

    model = make_ibm1(longest_tr=None if dynamic_sequence_length else experiment.training.longest_sequence(0),
                      longest_en=None if dynamic_sequence_length else experiment.training.longest_sequence(1),
                      vocab_size_tr= experiment.tks[0].vocab_size(),
                      vocab_size_en= experiment.tks[1].vocab_size(),
                      embedding_size=embedding_size,
                      optimizer=optimizer)
    # longest should become slightly more complicated: take from experiment, or
    # if dynamic sequence length is true stays none

    generator = BitextGenerator(experiment.tks[1].vocab_size(),
                            dynamic_sequence_length=dynamic_sequence_length)
    # TODO tks?

    experiment.run(model,
                   batch_size=batch_size,
                   nb_epochs=nb_epochs,
                   generator= generator,
                   early_stopping=early_stopping)
