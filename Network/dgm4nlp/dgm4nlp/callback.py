"""
:Authors: - Wilker Aziz
"""
import warnings
import numpy as np
import logging
import sys
import os
from itertools import cycle
from keras.callbacks import Callback
from dgm4nlp.nlputils import AERSufficientStatistics
from tabulate import tabulate
from keras import backend as K


def write_training_history(training_history, ostream=sys.stdout,
                           floatfmt='.4f', numalign='decimal', tablefmt='plain'):
    """
    Write training history (in terms of losses and metrics) to an output stream using a human-readable format.

    This function uses tabulate package.

    :param training_history: history as returned by keras.models.Model.fit
    :param ostream: output stream
    :param floatfmt:
    :param numalign:
    :param tablefmt:
    """
    # save history
    history_header = ['epoch']
    history_header.extend(sorted(training_history.history.keys()))
    history_data = []
    shift = 0
    for epoch in training_history.epoch:
        # fixes 0-based epochs
        if epoch == 0:
            shift = 1
        history_data.append([epoch + shift])
        for k, values in sorted(training_history.history.items(), key=lambda pair: pair[0]):
            history_data[-1].append(values[epoch])

    print(tabulate(history_data,
                   headers=history_header,
                   floatfmt=floatfmt,
                   numalign=numalign,
                   tablefmt=tablefmt),
          file=ostream)


class ViterbiAlignmentsWriter:
    """
    Wraps functions that write viterbi alignments to disk in different formats.

    """

    def __init__(self, output_dir, savefuncs):
        """

        :param output_dir: where to save files
        :param savefuncs: a dictionary mapping a format name to a function that prints the format
            each function takes (alignments, posterior, lengths, output_stream)
        """
        self._output_dir = output_dir
        self._savefuncs = savefuncs
        self._handlers = []
        os.makedirs(output_dir, exist_ok=True)

    def open(self, prefix=None):
        """
        Create an output file per format.
        :param prefix: an optional prefix
        """
        handlers = []
        for funcname, func in self._savefuncs.items():
            if prefix:
                filename = '{}/{}.{}'.format(self._output_dir, prefix, funcname)
            else:
                filename = '{}/{}'.format(self._output_dir, funcname)
            handlers.append(open(filename, 'w'))
        self._handlers = handlers

    def write(self, alignments, posteriors, lengths):
        """
        Save a batch of predictions.
        :param alignments:
        :param posteriors:
        :param lengths:
        :return:
        """
        for fo, (funcname, func) in zip(self._handlers, self._savefuncs.items()):
            func(alignments, posteriors, lengths, fo)

    def close(self):
        """
        Close all files.
        """
        for fo in self._handlers:
            fo.close()
        self._handlers = []


class EarlyStoppingChain(Callback):
    """Stop training after a sequence of one or more criteria are met.
    Criteria are considered in turn, that is, when criterion takes place after another has been met.
    Each criterion has to do with a monitored quantity no longer improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        mode: one of {min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing.
        initial_epoch: epoch at which we start monitoring quantities.
        verbose: verbosity mode.

    To specify multiple criteria simply provide lists for monitor, min_delta, patience and mode.
    """

    def __init__(self, monitor, min_delta, patience, mode, initial_epoch=1, verbose=False):
        super(EarlyStoppingChain, self).__init__()

        # normalise arguments
        if type(monitor) is str:
            monitor = [monitor]
        if type(min_delta) not in [list, tuple]:
            min_delta = [min_delta]
        if type(patience) is int:
            patience = [patience]
        if type(mode) is str:
            mode = [mode]

        # perform some sanity checks
        if not (len(monitor) == len(min_delta) == len(patience) == len(mode) > 0):
            raise ValueError("Bad configuration: monitor, min_delta, patience and mode should contain"
                             "the same number of elements and be non-empty.")
        if not all(m == 'min' or m == 'max' for m in mode):
            raise ValueError("Unknown modes in the list: %s" % mode)

        # these specify the convergence criteria
        self._monitor = monitor
        self._min_delta = min_delta
        self._patience = patience
        self._mode = mode
        self.initial_epoch = initial_epoch
        self.verbose = verbose
        self.shift = 0  # It seems like Keras counts epoch from 0 (here I am changing that)

        # these are used to log details about the convergence
        self._best = []
        self._stopped_epoch = []

        # these are used in order to configure the quantity we are currently monitoring
        self.criterion_iterator = None
        self.monitor = None
        self.monitor_op = None
        self.min_delta = None
        self.patience = None
        self.wait = None
        self.best = None

    def __repr__(self):
        return '%s.%s(monitor=%r, min_delta=%r, patience=%r, mode=%r, initial_epoch=%r, verbose=%r)' % (
            EarlyStoppingChain.__module__, EarlyStoppingChain.__name__,
            self._monitor, self._min_delta, self._patience, self._mode, self.initial_epoch, self.verbose)

    def next_criterion(self):
        """
        Configures the next convergence criterion.

        :return: False if we are done with convergence criteria.
        """
        try:
            self.monitor, self.min_delta, self.patience, mode = next(self.criterion_iterator)
            if mode == 'min':
                self.monitor_op = np.less
            else:
                self.monitor_op = np.greater

            if self.monitor_op == np.greater:
                self.min_delta *= 1
            else:
                self.min_delta *= -1
            self.wait = 0
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            if self.verbose:
                logging.info("Convergence criterion based on '%s'", self.monitor)
            return True
        except StopIteration:
            return False

    def on_train_begin(self, logs=None):
        # Construct a fresh iterator over criteria
        self.criterion_iterator = zip(self._monitor, self._min_delta, self._patience, self._mode)
        # Configure the first criterion
        self.next_criterion()

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.shift = 1

        epoch += self.shift  # this is to correct for 0-based epochs

        if epoch < self.initial_epoch:  # any convergence criterion will wait at least this many epochs
            return

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' % self.monitor, RuntimeWarning)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:  # if we run out of patience
                # we save details about the stopping point for criterion
                self._stopped_epoch.append(epoch)
                self._best.append(self.best)
                if not self.next_criterion():  # and if we run out of criteria, then we stop training
                    self.model.stop_training = True
            else:
                self.wait += 1

    def on_train_end(self, logs=None):
        if self.verbose:
            print("EarlyStoppingChain: monitor=%s stopped_epoch=%s best=%s" % (self._monitor,
                                                                               self._stopped_epoch,
                                                                               self._best))


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
        initial_epoch: epoch at which we start monitoring
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, initial_epoch=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.saved = []  # store (epoch, value, filepath) for saved models
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.initial_epoch = initial_epoch
        self.epochs_since_last_save = 0
        self.shift = 0  # It seems like Keras counts epoch from 0 (here I am changing that)

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.shift = 1
        epoch += self.shift  # this is to correct for 0-based epochs

        # check whether we are already monitoring
        if epoch < self.initial_epoch:
            return

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % self.monitor, RuntimeWarning)
                return

            if self.save_best_only:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
                    if self.saved:  # if we are saving the best only, we delete older files
                        os.unlink(self.saved[-1][2])
                        # and keep this one which has just been saved
                        self.saved[-1] = [epoch, current, filepath]
                    else:  # keep the first file saved
                        self.saved.append([epoch, current, filepath])
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                self.saved.append([epoch, current, filepath])


class ViterbiCallback(Callback):
    """
    A callback to decode with Viterbi algortihm.
    """

    def __init__(self, generator,
                 nb_steps,
                 viterbifunc,
                 viterbi_writer: ViterbiAlignmentsWriter=None,
                 save_freq=1,
                 initial_epoch=1,
                 name='viterbi'):
        """
        :param generator: an endless batch generator
        :param nb_steps: number of batches to be generated
        :param viterbifunc: a function that takes the inputs and predictions of the model
            and returns viterbi alignments and their posterior probabilities
        :param viterbi_writer: helper class to save viterbi alignments in different formats
        :param save_freq: Viterbi alignments will be saved on epochs multiple of this number
        :param initial_epoch: first epoch in which this callback applies
        :param name: callback name
        """
        super(ViterbiCallback, self).__init__()
        self._generator = generator
        self._steps = nb_steps
        self._viterbifunc = viterbifunc
        self._viterbi_writer = viterbi_writer
        self._save_freq = save_freq
        self._initial_epoch = initial_epoch
        self._name = name
        self.shift = 0

    def on_epoch_end(self, epoch, logs):
        # corrects 0-based epoch
        if epoch == 0:
            self.shift = 1
        epoch += self.shift

        # check whether it is too soon to decode
        if epoch < self._initial_epoch:
            return

        # if we are saving viterbi alignments
        if self._viterbi_writer and epoch % self._save_freq == 0:
            # open a file for each format
            self._viterbi_writer.open('epoch{:03d}'.format(epoch))
            saving = True
        else:
            saving = False

        for step, batch_info in enumerate(self._generator, 1):
            inputs = batch_info[0]
            predictions = self.model.predict_on_batch(inputs)
            alignments, posteriors = self._viterbifunc(inputs, predictions)
            lengths = np.not_equal(inputs[1], 0).sum(-1)  # x, y = inputs  (thus inputs[1])

            if saving:
                # save alignments
                self._viterbi_writer.write(alignments, posteriors, lengths)

            if step == self._steps:  # here we have already generated all steps
                break

        if saving:
            # close all handlers
            self._viterbi_writer.close()


class AERCallback(Callback):
    """
    A callback to decode with viterbi algorithm and compute AER.
    """

    def __init__(self, gold,
                 generator,
                 nb_steps,
                 viterbifunc,
                 history_filepath=None,
                 skip_null=True,
                 viterbi_writer: ViterbiAlignmentsWriter=None,
                 save_freq=1,
                 first_epoch=1,
                 name='aer'):
        """
        :param gold: gold-standard alignments
        :param generator: an endless batch generator
        :param nb_steps: number of batches to be generated
        :param viterbifunc: a function that takes the inputs and predictions of the model
            and returns viterbi alignments and their posterior probabilities
        :param history_filepath: where to save AER history
        :param skip_null: should we discard alignments to NULL
        :param viterbi_writer: helper class to save viterbi alignments in different formats
        :param save_freq: Viterbi alignments will be saved on epochs multiple of this number
        :param first_epoch: epoch at which we start decoding
        :param name: callback name
        """
        super(AERCallback, self).__init__()
        self._generator = generator
        self._steps = nb_steps
        self._gold = gold
        self._viterbifunc = viterbifunc
        self._history_filepath = history_filepath
        self._skip_null = skip_null
        self._viterbi_writer = viterbi_writer
        self._save_freq = save_freq
        self._first_epoch = first_epoch
        self._name = name
        self._history = []
        self.shift = 0

    def on_train_begin(self, logs=None):
        # TODO: check whether this is okay with Keras
        self.params['metrics'].append(self._name)

    def on_epoch_end(self, epoch, logs):
        # correct 0-based epoch
        if epoch == 0:
            self.shift = 1
        epoch += self.shift

        # check whether it is too soon to decode
        if epoch < self._first_epoch:
            return

        # decode
        gold_iterator = cycle(self._gold)
        suffstats = AERSufficientStatistics()

        # if we are saving viterbi alignments
        if self._viterbi_writer and epoch % self._save_freq == 0:
            # open a file for each format
            self._viterbi_writer.open('epoch{:03d}'.format(epoch))
            saving = True
        else:
            saving = False

        for step, batch_info in enumerate(self._generator, 1):
            inputs = batch_info[0]
            predictions = self.model.predict_on_batch(inputs)
            alignments, posteriors = self._viterbifunc(inputs, predictions)
            lengths = np.not_equal(inputs[1], 0).sum(-1)  # x, y = inputs  (thus inputs[1])

            if saving:
                # save alignments
                self._viterbi_writer.write(alignments, posteriors, lengths)

            for a, l, (sure, probable) in zip(alignments, lengths, gold_iterator):
                # we add 1 to j because labels are 1-based
                # we do not add 1 to a[j] because NULL token already sits at x[0]
                if self._skip_null:  # here we skip alignments to NULL
                    links = set((a[j], j + 1) for j in range(l) if a[j] > 0)  # we use a set of links format
                else:
                    links = set((a[j], j + 1) for j in range(l))  # we use a set of links format
                suffstats.update(sure, probable, links)

            if step == self._steps:
                break

        if saving:
            # close all handlers
            self._viterbi_writer.close()

        aer = suffstats.aer()
        self._history.append(aer)
        logs[self._name] = aer

    def on_train_end(self, logs=None):
        if self._history_filepath:
            with open(self._history_filepath, 'w') as fo:
                print('\n'.join('{0:.4f}'.format(aer) for aer in self._history), file=fo)
