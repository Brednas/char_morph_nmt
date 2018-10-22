"""
:Authors: - Wilker Aziz
"""
import os
os.environ["THEANO_FLAGS"] = "device=cuda1"

import numpy as np
import logging
import tempfile
import shutil
from datetime import datetime
from keras import optimizers
from dgm4nlp.embed.autoencoder import test_auto_encoder
from dgm4nlp.callback import EarlyStoppingChain

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

# 1. Get a unique working directory and save script for reproducibility
base_dir = '/home/wferrei1/experiments/embed/autoencoder/en-fr.en'
os.makedirs(base_dir, exist_ok=True)  # make sure base_dir exists
output_dir = tempfile.mkdtemp(prefix=datetime.now().strftime("%y-%m-%d.%Hh%Mm%Ss."), dir=base_dir)
logging.info('Workspace: %s', output_dir)
shutil.copy(os.path.abspath(__file__), output_dir)

# 2. Config convergence criteria
criteria = EarlyStoppingChain(monitor=['loss'],
                              min_delta=[0.1],
                              patience=[5],
                              mode=['min'],
                              initial_epoch=5,
                              verbose=True)

# 3. Train model
test_auto_encoder(
    # input/output paths
    training_path='/home/wferrei1/github/dgm4nlp/data/en-fr/training.en-fr.en',
    validation_path='/home/wferrei1/github/dgm4nlp/data/en-fr/trial.en-fr.en',
    output_dir='{}/training'.format(output_dir),
    # constraints
    vocab_size=30000,
    shortest=1,
    longest=50,
    dynamic_sequence_length=True,
    # architecture parameters
    emb_dim=128,
    emb_dropout=0.,
    hidden_layers=[(100, 'softplus')],
    # optimisation parameters
    batch_size=200,
    nb_epochs=10,
    optimizer=optimizers.Adagrad(),
    # convergence criteria
    early_stopping=criteria)
