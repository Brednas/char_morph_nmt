"""
:Authors: - Wilker Aziz
"""
import numpy as np
import os
from itertools import cycle
from dgm4nlp.nlputils import AERSufficientStatistics
from dgm4nlp.callback import ViterbiAlignmentsWriter


def viterbi(model, generator, nb_steps, viterbifunc, skip_null=True,
            viterbi_writer: ViterbiAlignmentsWriter=None,
            file_prefix=None):
    """
    Viterbi alignments from a model.

    :param model: trained model
    :param generator: test generator
    :param nb_steps: number of steps from generator
    :param viterbifunc: a viterbi algorithm compatible with the given model
    :param skip_null: whether we discard alignments to NULL
    :param viterbi_writer: helper class from writing viterbi alignments to disk
    :param file_prefix: a prefix for saved alignment files
    """
    if viterbi_writer:
        viterbi_writer.open(file_prefix)

    for step, batch_info in enumerate(generator, 1):
        inputs = batch_info[0]
        predictions = model.predict_on_batch(inputs)
        alignments, posteriors = viterbifunc(inputs, predictions)
        lengths = np.not_equal(inputs[1], 0).sum(-1)  # x, y = inputs  (thus inputs[1])
        if viterbi_writer:
            viterbi_writer.write(alignments, posteriors, lengths)
        if step == nb_steps:
            break

    if viterbi_writer:
        viterbi_writer.close()


def viterbi_aer(model, gold, generator, nb_steps, viterbifunc, skip_null=True,
                viterbi_writer: ViterbiAlignmentsWriter=None,
                file_prefix=None):
    """
    Compute AER for a test set and a trained model and save Viterbi alignments.

    :param model: trained model
    :param gold: gold standard alignment sets
    :param generator: test generator
    :param nb_steps: number of steps from generator
    :param viterbifunc: a viterbi algorithm compatible with the given model
    :param skip_null: whether we discard alignments to NULL
    :param viterbi_writer: helper class from writing viterbi alignments to disk
    :param file_prefix: a prefix for saved alignment files
    """
    gold_iterator = cycle(gold)
    suffstats = AERSufficientStatistics()

    if viterbi_writer:
        viterbi_writer.open(file_prefix)

    for step, batch_info in enumerate(generator, 1):
        inputs = batch_info[0]
        predictions = model.predict_on_batch(inputs)
        alignments, posteriors = viterbifunc(inputs, predictions)
        lengths = np.not_equal(inputs[1], 0).sum(-1)  # x, y = inputs  (thus inputs[1])

        if viterbi_writer:
            viterbi_writer.write(alignments, posteriors, lengths)

        for a, l, (sure, probable) in zip(alignments, lengths, gold_iterator):
            # we add 1 to j because labels are 1-based
            # we do not add 1 to a[j] because NULL token already sits at x[0]
            if skip_null:  # here we skip alignments to NULL
                links = set((a[j], j + 1) for j in range(l) if a[j] > 0)  # we use a set of links format
            else:
                links = set((a[j], j + 1) for j in range(l))  # we use a set of links format
            suffstats.update(sure, probable, links)

        if step == nb_steps:
            break

    if viterbi_writer:
        viterbi_writer.close()

    return suffstats.aer()
