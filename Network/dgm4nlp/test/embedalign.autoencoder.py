"""
:Authors: - Wilker Aziz
"""
import os
os.environ["THEANO_FLAGS"] = "device=cuda0"

import theano


import numpy as np
import logging
import tempfile
import shutil
from datetime import datetime
from keras import optimizers
from dgm4nlp.embedalign.autoencoder import test_auto_encoder
from dgm4nlp.callback import EarlyStoppingChain
from dgm4nlp.ibm import UniformAlignmentComponent

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

# 1. Get a unique working directory and save script for reproducibility
base_dir = '/home/wferrei1/experiments/embed-align/en-fr/ae/debug'
os.makedirs(base_dir, exist_ok=True)  # make sure base_dir exists
output_dir = tempfile.mkdtemp(prefix=datetime.now().strftime("%y-%m-%d.%Hh%Mm%Ss."), dir=base_dir)
logging.info('Workspace: %s', output_dir)
shutil.copy(os.path.abspath(__file__), output_dir)

# 2. Config convergence criteria
criteria = EarlyStoppingChain(monitor=['val_loss', 'val_aer'],
                              min_delta=[0.1, 0.01],
                              patience=[5, 5],
                              mode=['min', 'min'],
                              initial_epoch=10,
                              verbose=True)

# 3. Train model
test_auto_encoder(training_paths=['/home/wferrei1/github/dgm4nlp/data/en-fr/training.en-fr.en',
                                  '/home/wferrei1/github/dgm4nlp/data/en-fr/training.en-fr.fr'],
                  validation_paths=['/home/wferrei1/github/dgm4nlp/data/en-fr/trial.en-fr.en',
                                    '/home/wferrei1/github/dgm4nlp/data/en-fr/trial.en-fr.fr',
                                    '/home/wferrei1/github/dgm4nlp/data/en-fr/trial.en-fr.naacl'],
                  test_paths=['/home/wferrei1/github/dgm4nlp/data/en-fr/test.en-fr.en',
                              '/home/wferrei1/github/dgm4nlp/data/en-fr/test.en-fr.fr',
                              '/home/wferrei1/github/dgm4nlp/data/en-fr/test.en-fr.naacl'],
                  # data generation
                  vocab_size=[30000, 30000],
                  shortest=[1, 1],
                  longest=[40, 40],
                  dynamic_sequence_length=True,
                  # architecture parameters
                  emb_dim=256,
                  emb_dropout=0.,
                  nb_birnn_layers=0,
                  rnn_units=256,
                  rnn_recurrent_dropout=0.1,
                  rnn_dropout=0.1,
                  rnn_merge_mode='sum',
                  rnn_architecture='gru',
                  hidden_layers=[(256, 'relu')],
                  # alignment distribution options
                  alignment_component=UniformAlignmentComponent(),
                  # optimisation parameters
                  batch_size=200,
                  nb_epochs=100,
                  optimizer=optimizers.Adam(),
                  # convergence criteria
                  early_stopping=criteria,
                  # output files
                  output_dir='{}/training'.format(output_dir))
