"""
:Authors: - Adapted by Sander Bijl de Vroe, originally by Wilker Aziz
"""

#os.environ["THEANO_FLAGS"] = "device=cuda0"


#runs the graph/tests the model


import os
os.environ["THEANO_FLAGS"] = "device=cpu"

import theano


import numpy as np
import logging
import tempfile
import shutil
from datetime import datetime
from keras import optimizers
#from dgm4nlp.embedalign.autoencoder import test_auto_encoder
from dgm4nlp.callback import EarlyStoppingChain
from dgm4nlp.ibm import UniformAlignmentComponent
from ibm_model1 import make_ibm1, train_ibm1


tr_train = '/home/sander/Documents/Master/Thesis/data_may/smalltraintr'
en_train = '/home/sander/Documents/Master/Thesis/data_may/smalltrainen'
tr_val = '/home/sander/Documents/Master/Thesis/data_may/smalltesttr'
en_val = '/home/sander/Documents/Master/Thesis/data_may/smalltesten'

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')


base_dir = '/home/sander/Documents/Master/Thesis/experiments'
os.makedirs(base_dir, exist_ok=True)  # make sure base_dir exists
output_dir = tempfile.mkdtemp(prefix=datetime.now().strftime("%y-%m-%d.%Hh%Mm%Ss."), dir=base_dir)
logging.info('Workspace: %s', output_dir)
shutil.copy(os.path.abspath(__file__), output_dir)


criteria = EarlyStoppingChain(monitor='val_loss',
                              min_delta=0.1,
                              patience=5,
                              mode='min',
                              initial_epoch=10,
                              verbose=True)

train_ibm1([tr_train,en_train],
           [tr_val,en_val],
           [5000,5000],
           [1,1],
           longest=[20,20],
           dynamic_sequence_length=True,
           output_dir='{}/training'.format(output_dir),
           embedding_size = 256,
           batch_size = 100,
           nb_epochs = 20,
           optimizer = optimizers.Adam(),
           early_stopping = None)



# test_auto_encoder(training_paths=[tr_train,
#                                   en_train],
#                   validation_paths=[tr_val,
#                                     en_val],
#                   # test_paths=['/home/wferrei1/github/dgm4nlp/data/en-fr/test.en-fr.en',
#                   #             '/home/wferrei1/github/dgm4nlp/data/en-fr/test.en-fr.fr',
#                   #             '/home/wferrei1/github/dgm4nlp/data/en-fr/test.en-fr.naacl'],
#                   # data generation
#                   vocab_size=[30000, 30000],
#                   shortest=[1, 1],
#                   longest=[40, 40],
#                   dynamic_sequence_length=True,
#                   # architecture parameters
#                   emb_dim=256,
#                   emb_dropout=0.,
#                   nb_birnn_layers=0,
#                   rnn_units=256,
#                   rnn_recurrent_dropout=0.1,
#                   rnn_dropout=0.1,
#                   rnn_merge_mode='sum',
#                   rnn_architecture='gru',
#                   hidden_layers=[(256, 'relu')],
#                   # alignment distribution options
#                   alignment_component=UniformAlignmentComponent(),
#                   # optimisation parameters
#                   batch_size=200,
#                   nb_epochs=100,
#                   optimizer=optimizers.Adam(),
#                   # convergence criteria
#                   early_stopping=criteria,
#                   # output files
#                   output_dir='{}/training'.format(output_dir))
