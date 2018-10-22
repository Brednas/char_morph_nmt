"""
:Authors: - Wilker Aziz, adapted for characters and morphemes by Sander Bijl de Vroe
"""
import os
os.environ["THEANO_FLAGS"] = "device=cpu"

import theano

import numpy as np
import logging
import tempfile
import shutil
from datetime import datetime
from keras import optimizers
from dgm4nlp.embedalign_morph.morph_char_autoencoder import test_morph_auto_encoder
from dgm4nlp.callback import EarlyStoppingChain
from dgm4nlp.ibm import UniformAlignmentComponent

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

# 1. Get a unique working directory and save script for reproducibility
#base_dir = '/mnt/data/sander/experiments'
base_dir = '/home/sander/Documents/Master/Thesis/experiments'
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
test_morph_auto_encoder(training_paths=['/home/sander/Documents/Master/Thesis/data_may/tr500',
                                    '/home/sander/Documents/Master/Thesis/data_may/en500',
                                    '/home/sander/Documents/Master/Thesis/data_may/mo500'],
                  validation_paths=['/home/sander/Documents/Master/Thesis/data_may/tr500',
                                    '/home/sander/Documents/Master/Thesis/data_may/en500',
                                    '/home/sander/Documents/Master/Thesis/data_may/mo500'],
                  # data generation
                  vocab_size=[1000, 30000, 500],
                  x_char_vocab_size = 500,
                  shortest=[1, 1, 1],
                  longest=[40, 40, 40],
                  dynamic_sequence_length=True,
                  # architecture parameters
                  char_emb_dim=256,
                  num_filters=[150,93,12],
                  morph_emb_size = 64,
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
                  nb_epochs=3,
                  optimizer=optimizers.Adam(),
                  # convergence criteria
                  early_stopping=criteria,
                  # output files
                  output_dir='{}/training'.format(output_dir))


# test_morph_auto_encoder(training_paths=['/mnt/data/sander/data/tr.norm.top205677train',
#                                   '/mnt/data/sander/data/en.norm.top205677train',
#                                        '/mnt/data/sander/data/tr.morph.top205677train'],
#                   validation_paths=['/mnt/data/sander/data/tr.norm.mid1000val',
#                                     '/mnt/data/sander/data/en.norm.mid1000val',
#                                     '/mnt/data/sander/data/tr.morph.mid1000val',
#                                     '/mnt/data/sander/data/shuffled/val1test23/word.en-tr.naacl.val'],
#                   test_paths=['/mnt/data/sander/data/tr.norm.bot1000test',
#                               '/mnt/data/sander/data/en.norm.bot1000test',
#                                 '/mnt/data/sander/data/tr.morph.bot1000test'
#                               '/mnt/data/sander/data/shuffled/val1test23/word.en-tr.naacl.test'],
