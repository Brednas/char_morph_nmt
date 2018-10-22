"""
:Authors: - Wilker Aziz, adapted for characters and morphemes by Sander Bijl de Vroe
"""
import logging
import numpy as np
import os
from time import time
from keras.models import Model
from keras import backend as K
from dgm4nlp.recipes import smart_ropen
from dgm4nlp.blocks import Generator
from dgm4nlp.lemutils import Multitext
from dgm4nlp.nlputils import read_naacl_alignments
from dgm4nlp.nlputils import save_moses_alignments
from dgm4nlp.callback import ModelCheckpoint
from dgm4nlp.callback import AERCallback
from dgm4nlp.callback import ViterbiCallback
from dgm4nlp.callback import write_training_history
from dgm4nlp.callback import ViterbiAlignmentsWriter
from dgm4nlp.embedalign.decode import viterbi
from dgm4nlp.embedalign.decode import viterbi_aer
from tabulate import tabulate


class ExperimentWrapper:

    def __init__(self, training_paths: list,
                 validation_paths: list,
                 tok_classes: list,
                 # data pre-processing
                 nb_words: list, # first: max number of characters(none), second max words in y vocab, third max morphs
                 shortest_sequence: list,
                 longest_sequence: list,
                 # output
                 output_dir: str,
                 # padding
                 bos_str=[None, None],
                 dynamic_sequence_length=False,
                 test_paths: list=[]):
        """

        :param training_paths:
        :param validation_paths:
        :param nb_words:
        :param shortest_sequence:
        :param longest_sequence:
        """
        assert len(training_paths) == 3, 'I expect a bitext and corresponding lemmas'

        # prepare data
        nb_streams = len(training_paths)
        logging.info('Fitting vocabularies')
        tks = []
        for i, (path, tok_class, vs, bos) in enumerate(zip(training_paths, tok_classes, nb_words, bos_str)):
                logging.info(' stream=%d', i)
                # tokenizer with a bounded vocabulary
                tks.append(tok_class(nb_words=vs, bos_str=bos))
                tks[-1].fit_one(smart_ropen(path))
                logging.info('  vocab-size=%d', tks[-1].vocab_size())

        logging.info('Memory mapping training data')

        training = Multitext(training_paths,
                             tokenizers=tks,
                             shortest=shortest_sequence,
                             longest=longest_sequence,
                             trim=[dynamic_sequence_length] * nb_streams,
                             mask_dtype=K.floatx(),
                             name='training')

        # in case the longest sequence was shorter than we thought
        longest_sequence = [training.longest_sequence(i) for i in range(nb_streams)]
        logging.info(' training-samples=%d longest=%s tokens=%s', training.nb_samples(),
                     longest_sequence,
                     [training.nb_tokens(i) for i in range(nb_streams)])

        if validation_paths:
            logging.info('Memory mapping validation data')
            validation = Multitext(validation_paths[0:3],
                                   tokenizers=tks,
                                   shortest=shortest_sequence,
                                   longest=longest_sequence,
                                   trim=[dynamic_sequence_length] * nb_streams,
                                   mask_dtype=K.floatx(),
                                   name='validation')
            logging.info(' dev-samples=%d', validation.nb_samples())
            if len(validation_paths) == 4:  # we have a NAACL file for alignments
                logging.info("Working with gold labels for validation: '%s'", validation_paths[3])
                # reads in sets of gold alignments
                val_gold_alignments = read_naacl_alignments(validation_paths[3])
                # discard those associated with sentences that are no longer part of the validation set
                #  (for example due to length constraints)
                val_gold_alignments = [a_sets for keep, a_sets in zip(validation.iter_selection_flags(),
                                                                  val_gold_alignments) if keep]
                logging.info(' gold-samples=%d', len(val_gold_alignments))
            else:
                val_gold_alignments = None
        else:
            validation = None
            val_gold_alignments = None

        if test_paths:
            logging.info('Memory mapping test data')
            test = Multitext(test_paths[0:3],
                             tokenizers=tks,
                             shortest=None,
                             longest=None,
                             trim=None,
                             mask_dtype=K.floatx(),
                             name='test')
            logging.info(' test-samples=%d', test.nb_samples())
            if len(test_paths) == 4:  # we have a NAACL file for alignments
                logging.info("Working with gold labels for test: '%s'", test_paths[-1])
                # reads in sets of gold alignments
                test_gold_alignments = read_naacl_alignments(test_paths[-1])
                # discard those associated with sentences that are no longer part of the validation set
                #  (for example due to length constraints)
                test_gold_alignments = [a_sets for keep, a_sets in zip(test.iter_selection_flags(),
                                                                       test_gold_alignments) if keep]
                logging.info(' test-gold-samples=%d', len(test_gold_alignments))
            else:
                test_gold_alignments = None
        else:
            test = None
            test_gold_alignments = None

        # create output directory
        os.makedirs(output_dir, exist_ok=True)

        self.tks = tks
        self.training = training
        self.validation = validation
        self.val_gold_alignments = val_gold_alignments
        self.output_dir = output_dir
        self.test = test
        self.test_gold_alignments = test_gold_alignments

    def save_training_history(self, training_history):
        path = '{}/history'.format(self.output_dir)
        with open(path, 'w') as fh:
            write_training_history(training_history, ostream=fh, tablefmt='plain')
        return path

    def get_checkpoint_callback(self, monitor, mode='min'):
        """Return a ModelCheckpoint callback object that monitors a certain quantity"""
        # make sure the directory exists
        os.makedirs('%s/weights' % self.output_dir, exist_ok=True)
        # get a template name for weight file
        model_file = '%s/weights/epoch={epoch:03d}.%s={%s:.4f}.hdf5' % (self.output_dir, monitor, monitor)
        # return a model checkpoint for that monitor
        return ModelCheckpoint(filepath=model_file,
                               monitor=monitor,
                               mode=mode,
                               save_weights_only=True,
                               save_best_only=True)

    def run(self, model: Model,
            batch_size,
            nb_epochs,
            generator: Generator,
            viterbifunc=None,
            early_stopping=None):
        """

        :param model: an embed-align Model
        :param batch_size: samples per batch
        :param nb_epochs: number of epochs
        :param generator: embed-align batch generator (compatible with your choice of model)
        :param viterbifunc: a function to decode viterbi alignments (again, compatible with your choice of model)
        :param early_stopping: a callback for early stopping
        :return:
        """
        t0 = time()

        logging.info('Starting training')
        callbacks = []

        # create a Viterbi writer for validation experiments
        val_viterbi_writer = ViterbiAlignmentsWriter(output_dir='{}/viterbi-{}'.format(self.output_dir,
                                                                                       self.validation.name),
                                                     savefuncs={'moses': save_moses_alignments})

        if self.val_gold_alignments and viterbifunc:  # in case we have gold alignments we add an AER callback
            callbacks.append(AERCallback(gold=self.val_gold_alignments,
                                         generator=generator.get(self.validation,
                                                                 batch_size=batch_size),
                                         nb_steps=np.ceil(self.validation.nb_samples() / batch_size),
                                         viterbifunc=viterbifunc,
                                         history_filepath=None if not self.output_dir else '{}/viterbi-{}/history'.format(
                                             self.output_dir,
                                             self.validation.name),
                                         viterbi_writer=val_viterbi_writer,
                                         name='val_aer'))

            # because AER is available we use it to save model weights
            callbacks.append(self.get_checkpoint_callback('val_aer', 'min'))

        elif viterbifunc:  # otherwise we simply compute Viterbi alignments
            callbacks.append(ViterbiCallback(generator.get(self.validation,
                                                           batch_size=batch_size),
                                             nb_steps=np.ceil(self.validation.nb_samples() / batch_size),
                                             viterbifunc=viterbifunc,
                                             viterbi_writer=val_viterbi_writer,
                                             name='val_viterbi'))

        # by default, we also save models depending on training and validation loss
        callbacks.append(self.get_checkpoint_callback('loss', 'min'))
        callbacks.append(self.get_checkpoint_callback('val_loss', 'min'))

        if early_stopping:  # users may configure early stopping criteria
            callbacks.append(early_stopping)

        K.set_learning_phase(1)

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

        # log training time
        dt = time() - t0
        with open('{}/time.secs'.format(self.output_dir), 'w') as fo:
            print('training=%s' % dt, file=fo)
        t0 = time()

        if not self.test:
            return
        # We have a test set!
        logging.info('Testing selected models')

        # get a viterbi writer
        test_viterbi_writer = ViterbiAlignmentsWriter(output_dir='{}/viterbi-{}'.format(self.output_dir,
                                                                                        self.test.name),
                                                      savefuncs={'moses': save_moses_alignments})

        # test selected models
        test_results = []
        for callback in callbacks:
            if type(callback) is not ModelCheckpoint:
                continue
            if not callback.saved:
                continue
            # we test only the best model for each monitored quantity
            monitor = callback.monitor
            epoch, performance, saved_weights = callback.saved[-1]
            print('Loading: %s' % saved_weights)
            # load the weights
            model.load_weights(saved_weights)
            if not self.test_gold_alignments:  # test without gold alignments
                viterbi(model,
                        generator.get(self.test,
                                      batch_size=batch_size),
                        nb_steps=np.ceil(self.test.nb_samples() / batch_size),
                        viterbifunc=viterbifunc,
                        skip_null=True,
                        viterbi_writer=test_viterbi_writer,
                        file_prefix='{}-{}'.format(monitor, epoch))
            else:  # test and compute AER
                aer = viterbi_aer(model,
                                  self.test_gold_alignments,
                                  generator.get(self.test,
                                                batch_size=batch_size),
                                  nb_steps=np.ceil(self.test.nb_samples() / batch_size),
                                  viterbifunc=viterbifunc,
                                  skip_null=True,
                                  viterbi_writer=test_viterbi_writer,
                                  file_prefix='{}-{}'.format(monitor, epoch))
                print('AER %s' % aer)
                test_results.append([monitor, epoch, performance, aer])

        # Save test evaluation results
        if test_results:
            with open('{}/viterbi-{}/results.txt'.format(self.output_dir, self.test.name), 'w') as fo:
                print(tabulate(test_results,
                               headers=['monitor', 'epoch', 'training-performance', 'test_aer'],
                               floatfmt='.4f', numalign='decimal', tablefmt='plain'),
                      file=fo)

        dt = time() - t0
        with open('{}/time.secs'.format(self.output_dir), 'a') as fo:
            print('test=%s' % dt, file=fo)
