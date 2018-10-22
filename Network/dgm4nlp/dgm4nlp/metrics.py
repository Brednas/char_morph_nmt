"""
Metrics specialised for sequence tasks where time steps are not independent
    * For losses, Keras takes means along time dimensions, we typically want to sum
    * Keras normalises by mean of masks, we typically want to normalise by sum across time dimension

:Authors: - Wilker Aziz
"""
from keras import backend as K


def accuracy(y_true, y_pred, masked_value=0):
    """
    Accuracy where some samples are masked
    :param y_true: one-hot labels (sample dimension, (time dimension,) label dimension)
    :param y_pred: predicted distribution (sample dimension, (time dimension,) label dimension)
    :param masked_value: masked label (defaults to 0)
    :return: accuracy discarding masked labels
    """
    # gold labels
    # (B, M)
    gold = K.argmax(y_true, axis=-1)        # e.g. 0 1 2 3 0 0 0
    # predicted labels
    # (B, M)
    pred = K.argmax(y_pred, axis=-1)        # e.g. 1 0 1 1 1 0 1
    # masked positions
    # (B, M)
    mask = K.cast(K.not_equal(gold, masked_value), K.floatx())  # e.g. 0 1 1 1 0 0 0
    # compare and mask
    # (B, M)
    cmp = K.cast(K.equal(gold, pred), K.floatx()) * mask        # e.g. 0 0 1 1 0 0 0
    # (B,)
    # get total (discarding masked labels)
    total = mask.sum(axis=-1)                      # e.g. 3
    # accuracy
    # (B,)
    return cmp.sum(axis=-1) / total  # average along the time dimension


def categorical_crossentropy(y_true, y_pred, masked_value=0):
    """

    :param y_true: one-hot gold vectors
    :param y_pred: distribution over the output classes
        Note that labels.shape == distribution.shape
    :param masked_value: -PAD- label (defaults to 0)
    :return: - \sum p(x) log q(x)
    """

    # (B, M)
    mask = K.cast(K.not_equal(K.argmax(y_true, axis=-1),
                              masked_value), K.floatx())

    # (B * M,)
    minus_loglikelihood = K.categorical_crossentropy(
        # reshape into (B * M, V)
        K.reshape(y_pred, (-1, y_pred.shape[-1])),
        # reshape into (B * M, V)
        K.reshape(y_true, (-1, y_pred.shape[-1])))
    # (B, M)
    minus_loglikelihood = K.reshape(minus_loglikelihood, (y_pred.shape[0], y_pred.shape[1]))

    # maximise(-KL + E) becomes minimise(-E + KL)
    # (B, M)
    loss = minus_loglikelihood * mask
    #total = mask.sum(-1)
    # (B,)
    return loss.sum(-1) # sum along the time dimension

def categorical_crossentropy_t4(y_true, y_pred, masked_value=0):
    """

    :param y_true: one-hot gold vectors
    :param y_pred: distribution over the output classes
        Note that labels.shape == distribution.shape
    :param masked_value: -PAD- label (defaults to 0)
    :return: - \sum p(x) log q(x)
    """

    # (B, M, Mph)
    mask = K.cast(K.not_equal(K.argmax(y_true, axis=-1),
                              masked_value), K.floatx())

    # (B * Mr)
    minus_loglikelihood = K.categorical_crossentropy(
        # reshape into (B * M, V)
        K.reshape(y_pred, (-1, y_pred.shape[-1])),
        # reshape into (B * M, V)
        K.reshape(y_true, (-1, y_pred.shape[-1])))
    # (B, M)
    minus_loglikelihood = K.reshape(minus_loglikelihood, (y_pred.shape[0], y_pred.shape[1]))

    # maximise(-KL + E) becomes minimise(-E + KL)
    # (B, M)
    loss = minus_loglikelihood * mask
    #total = mask.sum(-1)
    # (B,)
    return loss.sum(-1) # sum along the time dimension


def mask_kl(mask, kl):
    """
    Mask and aggregate KL values in a batch of sequences.

    :param mask: a (B, T) mask for KL values
    :param kl: (B, T) KL values
    :return: sum along time dimension of masked KL values
    """
    return K.sum(kl * mask, axis=-1)  #  we sum along the time dimension / mask.sum(-1)


def masked_categorical_crossentropy(mask):
    def f(y_true, y_pred):
        """

        :param y_true: one-hot gold vectors
        :param y_pred: distribution over the output classes
            Note that labels.shape == distribution.shape
        :param masked_value: -PAD- label (defaults to 0)
        :return: - \sum p(x) log q(x)
        """
        # (B * M,)
        minus_loglikelihood = K.categorical_crossentropy(
            # reshape into (B * M, V)
            K.reshape(y_pred, (-1, y_pred.shape[-1])),
            # reshape into (B * M, V)
            K.reshape(y_true, (-1, y_pred.shape[-1])))
        # (B, M)
        minus_loglikelihood = K.reshape(minus_loglikelihood, (y_pred.shape[0], y_pred.shape[1]))

        # maximise(-KL + E) becomes minimise(-E + KL)
        # (B, M)
        loss = minus_loglikelihood * mask
        # (B,)
        return loss.sum(-1)
    return f