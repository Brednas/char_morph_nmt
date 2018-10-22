"""
:Authors: - Wilker Aziz
"""
import os
import logging
import numpy as np
from keras.models import Model
from dgm4nlp.nlputils import Tokenizer
from dgm4nlp.nlputils import Text
from dgm4nlp.recipes import smart_ropen
from dgm4nlp.blocks import Generator
from dgm4nlp.callback import write_training_history
from dgm4nlp.callback import ModelCheckpoint


class ExperimentWrapper:
    def __init__(self, training_path, validation_path,
                 # data pre-processing
                 nb_words,
                 shortest_sequence,
                 longest_sequence,
                 # output files
                 output_dir,
                 # dynamic dimensions
                 dynamic_sequence_length=False):

        # prepare data
        logging.info('Fitting vocabulary')
        tk = Tokenizer(nb_words=nb_words)  # tokenizer with a bounded vocabulary
        tk.fit_one(smart_ropen(training_path))
        logging.info(' vocab-size=%d' % tk.vocab_size())
        logging.info('Memory mapping training data')
        training = Text(training_path, tk,
                        shortest=shortest_sequence,
                        longest=longest_sequence,
                        trim=dynamic_sequence_length)
        longest_sequence = training.longest_sequence()  # in case the longest sequence was shorter than we thought
        logging.info(' training-samples=%d longest=%d tokens=%d', training.nb_samples(),
                     training.longest_sequence(),
                     training.nb_tokens())

        if validation_path:
            logging.info('Memory mapping validation data')
            validation = Text(validation_path, tk,
                              shortest=shortest_sequence,
                              longest=longest_sequence,
                              trim=dynamic_sequence_length)
            logging.info(' dev-samples=%d', validation.nb_samples())
        else:
            validation = None

        os.makedirs(output_dir, exist_ok=True)

        self.tk = tk
        self.training = training
        self.validation = validation
        self.output_dir = output_dir

    def get_checkpoint_callback(self, monitor, mode='min'):
        """Return a ModelCheckpoint callback object that monitors a certain quantity"""
        model_file = '%s/weights.epoch={epoch:03d}.%s={%s:.4f}.hdf5' % (self.output_dir, monitor, monitor)
        return ModelCheckpoint(filepath=model_file,
                               monitor=monitor,
                               mode=mode,
                               save_weights_only=True,
                               save_best_only=True)

    def save_training_history(self, training_history):
        path = '{}/history'.format(self.output_dir)
        with open(path, 'w') as fh:
            write_training_history(training_history, ostream=fh, tablefmt='plain')
        return path

    def fit(self, model: Model,
            batch_size,
            nb_epochs,
            generator: Generator,
            early_stopping=None):

        logging.info('Starting training')

        # by default we save models based on training/validation loss
        callbacks = [self.get_checkpoint_callback('loss', 'min'),
                     self.get_checkpoint_callback('val_loss', 'min')]

        if early_stopping:
            callbacks.append(early_stopping)

        history = model.fit_generator(generator.get(self.training,
                                                    batch_size=batch_size),
                                      steps_per_epoch=np.ceil(self.training.nb_samples() / batch_size),
                                      validation_data=generator.get(self.validation,
                                                                    batch_size=batch_size) if self.validation else None,
                                      validation_steps=np.ceil(
                                          self.validation.nb_samples() / batch_size) if self.validation else None,
                                      epochs=nb_epochs,
                                      callbacks=callbacks)

        self.save_training_history(history)
        logging.info('Check output files in: %s', self.output_dir)
