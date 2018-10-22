"""
:Authors: - Wilker Aziz, adapted for characters and morphemes by Sander Bijl de Vroe
"""
import os
os.environ["THEANO_FLAGS"] = "device=cpu"

import numpy as np
import logging
import tempfile
import shutil
from datetime import datetime
from dgm4nlp.linguistic_encoder_models.morph_ae import test_auto_encoder
from dgm4nlp.callback import EarlyStoppingChain
from dgm4nlp.ibm import UniformAlignmentComponent

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

#TODO change char embedding dim

# 1. Get a unique working directory and save script for reproducibility
base_dir = '~/Documents/Master/Thesis/experiments'
os.makedirs(base_dir, exist_ok=True)  # make sure base_dir exists
output_dir = tempfile.mkdtemp(prefix = 'morphlem', suffix=datetime.now().strftime("%y-%m-%d.%Hh%Mm%Ss."), dir=base_dir)
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
test_auto_encoder(training_paths=['/home/sander/Documents/Master/Thesis/data_may/tr.norm',
                                  '/home/sander/Documents/Master/Thesis/data_may/en.norm',
                                  '/home/sander/Documents/Master/Thesis/data_may/tr.morph'],
                  validation_paths=['/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split1/word.en-tr.tr1',
                                    '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split1/word.en-tr.en1',
                                    '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split1/word.en-tr.mo1',
                                    '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split1/word.en-tr.naacl1'],
                  test_paths=['/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split23/word.en-tr.tr23',
                              '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split23/word.en-tr.en23',
                              '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split23/word.en-tr.mo23',
                              '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split23/word.en-tr.naacl23'],
                            # data generation
                            vocab_size=[500, 30000, 500],
                            x_char_vocab_size = 500,
                            x_morph_vocab_size = 500,
                            shortest=[1, 1, 1],
                            longest=[40, 40, 40],
                            dynamic_sequence_length=True,
                            # architecture parameters
                            char_emb_dim=128,
                            morph_emb_dim=128,
                            num_filters = [75,47,6],
                            cnn_activation = 'tanh',
                            rnn_units=128,
                            rnn_dropout=0,
                            rnn_recurrent_dropout=0,
                            rnn_merge_mode='sum',
                            hidden_layers=[(128, 'relu')],
                            # alignment distribution options
                            alignment_component=UniformAlignmentComponent(),
                            # optimisation parameters
                            batch_size=64,
                            nb_epochs=100,
                            # convergence criteria
                            early_stopping=criteria,
                            # output files
                            output_dir='{}/training'.format(output_dir))




#
# """
# :Authors: - Wilker Aziz
# """
# import os
# os.environ["THEANO_FLAGS"] = "device=cpu"
#
# import numpy as np
# import logging
# import tempfile
# import shutil
# from datetime import datetime
# from dgm4nlp.linguistic_encoder_models.morphlem_ae import test_auto_encoder
# from dgm4nlp.callback import EarlyStoppingChain
# from dgm4nlp.ibm import UniformAlignmentComponent
#
# np.random.seed(42)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
#
# #TODO change char embedding dim
#
# # 1. Get a unique working directory and save script for reproducibility
# base_dir = '~/Documents/Master/Thesis/experiments'
# os.makedirs(base_dir, exist_ok=True)  # make sure base_dir exists
# output_dir = tempfile.mkdtemp(prefix = 'morphlem', suffix=datetime.now().strftime("%y-%m-%d.%Hh%Mm%Ss."), dir=base_dir)
# logging.info('Workspace: %s', output_dir)
# shutil.copy(os.path.abspath(__file__), output_dir)
#
# # 2. Config convergence criteria
# criteria = EarlyStoppingChain(monitor=['val_loss', 'val_aer'],
#                               min_delta=[0.1, 0.01],
#                               patience=[5, 5],
#                               mode=['min', 'min'],
#                               initial_epoch=10,
#                               verbose=True)
#
#
#
#
# # 3. Train model
# test_auto_encoder(training_paths=['/home/sander/Documents/Master/Thesis/data_may/tr.norm',
#                                   '/home/sander/Documents/Master/Thesis/data_may/en.norm',
#                                   '/home/sander/Documents/Master/Thesis/data_may/tr.morph',
#                                   '/home/sander/Documents/Master/Thesis/data_may/tr.morph'],
#                   validation_paths=['/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split1/word.en-tr.tr1',
#                                     '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split1/word.en-tr.en1',
#                                     '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split1/word.en-tr.mo1',
#                                     '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split1/word.en-tr.mo1',
#                                     '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split1/word.en-tr.naacl1'],
#                   test_paths=['/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split23/word.en-tr.tr23',
#                               '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split23/word.en-tr.en23',
#                               '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split23/word.en-tr.mo23',
#                               '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split23/word.en-tr.mo23',
#                               '/home/sander/Documents/Master/Thesis/data_may/en-tr/shuffled/split23/word.en-tr.naacl23'],
#                             # data generation
#                             vocab_size=[500, 30000, 500, 30000],
#                             x_char_vocab_size = 500,
#                             x_morph_vocab_size = 500,
#                             shortest=[1, 1, 1, 1],
#                             longest=[40, 40, 40, 40],
#                             dynamic_sequence_length=True,
#                             # architecture parameters
#                             char_emb_dim=128,
#                             lem_emb_dim=128,
#                             morph_emb_dim=128,
#                             num_filters = [75,47,6],
#                             cnn_activation = 'tanh',
#                             rnn_units=128,
#                             rnn_dropout=0,
#                             rnn_recurrent_dropout=0,
#                             rnn_merge_mode='sum',
#                             hidden_layers=[(128, 'relu')],
#                             # alignment distribution options
#                             alignment_component=UniformAlignmentComponent(),
#                             # optimisation parameters
#                             batch_size=64,
#                             nb_epochs=100,
#                             # convergence criteria
#                             early_stopping=criteria,
#                             # output files
#                             output_dir='{}/training'.format(output_dir))
