"""
Here I have lots of code for typical building blocks.
 - I avoid writing a layer class whenever a Lambda layer will do.
 - This code is taylored for batches of sequences.

:Authors: - Wilker Aziz
"""
from keras import layers
from keras import backend as K


class Generator:

    def __init__(self, shorter_batch='trim',
                 endless=True,
                 dynamic_sequence_length=False):
        """

        :param shorter_batch: what to do with shorter batches (defaults to 'trim')
        :param endless: generated endlessly (defaults to True)
        :param dynamic_sequence_length:  whether we trim time steps (defaults to False)
        """
        self.shorter_batch = shorter_batch
        self.endless = endless
        self.dynamic_sequence_length = dynamic_sequence_length

    def get(self, corpus, batch_size):
        pass


def Identity(shape, dtype=None, name='Identity'):
    """
    Syntactic sugar for an identity lambda layer
        - useful for turning Input into an outputable layer

    Example: Identity(shape=(longest,), dtype=K.floatx())(x)

    :param shape: input shape (without the sample dimension)
    :param dtype: if None, maintains input type, otherwise casts to dtype
    :param name:
    """
    def instantiate(inputs):
        if dtype is None:
            return layers.Lambda(lambda t: t,
                                 output_shape=shape,
                                 name=name)(inputs)
        else:
            return layers.Lambda(lambda t: K.cast(t, dtype=dtype),
                                 output_shape=shape,
                                 name=name)(inputs)
    return instantiate


def MakeMask(shape, masked_value=0, dtype=K.floatx(), name=None):
    """
    Construct a mask based on a -PAD- value.

    Example:
        GetMask(name='mask')(x)
    :param masked_value: defaults to 0
    :param dtype: if None, returns booleans, otherwise casts to dtype
        this defaults to K.floatx() because that's the typical use of masks
    :param name:
    """
    def instantiate(x):
        if dtype is None:
            return layers.Lambda(lambda t: K.not_equal(x, masked_value),
                                 output_shape=shape,
                                 name=name)(x)
        else:
            return layers.Lambda(lambda t: K.cast(K.not_equal(x, masked_value), K.floatx()),
                                 output_shape=shape,
                                 name=name)(x)
    return instantiate


def ApplyMask(shape, broadcast_axis=None, name='ApplyMask'):
    """
    Syntactic sugar for a lambda layer that masks a tensor using a bixnary squared matrix.

    Example:
        ApplyMask(shape=(longest_y, longest_x), axis=1)([pa_x, x_mask])
        ApplyMask(shape=(longest_x, longest_y), axis=2)([py_z, y_mask])

    :param shape: output shape (no need to specify the sample dimension)
    :param broadcast_axis: either 1 (for time dimension of the first stream)
        or 2 (for the time dimension of the second stream).
        It defaults to None, in which case no broadcasting happens.
    :param name:
    """
    def instantiate(args):
        inputs, mask = args
        if broadcast_axis is None:
            return layers.Lambda(lambda pair: pair[0] * pair[1],
                                 output_shape=shape,
                                 name=name)([inputs, mask])
        elif broadcast_axis == 1:
            return layers.Lambda(lambda pair: pair[0] * pair[1][:, None, :],
                                 output_shape=shape,
                                 name=name)([inputs, mask])
        elif broadcast_axis == 2:
            return layers.Lambda(lambda pair: pair[0] * pair[1][:, :, None],
                                 output_shape=shape,
                                 name=name)([inputs, mask])
        else:
            raise ValueError('broadcast_axis can be None, 1 or 2, got %d' % broadcast_axis)
    return instantiate


def kl_from_q_to_standard_normal(args):
    """
    KL between q and p where
        q is N(mean, var)
        p is N(0, 1)
    :param args: [mean, log_var]
    :return: KL
    """
    mean, log_var = args
    return -0.5 * K.sum(log_var - K.exp(log_var) - K.square(mean) + 1, axis=-1)


def kl_diagonal_gaussians(args):
    """
    KL between q and p where
        q is N(mean1, var1)
        p is N(mean2, var2)

    References:
        - https://tgmstat.wordpress.com/2013/07/10/kullback-leibler-divergence/
        - https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Kullback.E2.80.93Leibler_divergence_for_multivariate_normal_distributions
        - https://arxiv.org/pdf/1611.01437.pdf

    :param args: [mean1, log_var1, mean2, log_var2]
    :return: KL
    """
    mean1, log_var1, mean2, log_var2 = args
    var1 = K.exp(log_var1)
    var2 = K.exp(log_var2)
    return 0.5 * K.sum(log_var2 - log_var1 + (var1 + K.square(mean1 - mean2)) / var2 - 1, axis=-1)


def gaussian_noise(noise_mean=0., noise_std=1.):
    """
    Standard Gaussian reparameterisation.

    Exampe: layers.Lambda(gaussian_noise(0, 1), output_shape=(longest, dz))([mean, log_var])

    :param noise_mean: defaults to 0
    :param noise_std: defaults to 1
    :return: mean + std * epsilon
        where epsilon ~ N(0, 1)
    """
    def instantiate(args):
        mean, log_var = args
        epsilon = K.random_normal(mean.shape, noise_mean, noise_std)
        return mean + K.exp(log_var / 2.) * epsilon
    return instantiate


def SampleFromGaussian(shape, noise_mean=0., noise_std=1., name='gaussian-sample'):
    """
    Sample from a parameterised Gaussian.

    Example: SampleFromGaussian(shape=(longest, dz))([mean, log_var])

    :param shape: (longest, units)
    :param noise_mean: defaults to 0
    :param noise_std: defaults to 1
    :param name:
    """
    def instantiate(args):
        mean, log_var = args
        return layers.Lambda(gaussian_noise(noise_mean=noise_mean, noise_std=noise_std),
                             output_shape=shape,
                             name=name)([mean, log_var])
    return instantiate


def GaussianPrior(shape, fit_mean, fit_var, name='GaussianPrior'):
    """
    Constructs a Gaussian prior compatible with a certain shape.

    Example: GaussianPrior(shape=(longest, dz), fit_mean=False, fit_prior=True)(z)

        We give z as input so we can produce zeros like it (I haven't found a better way in Keras yet).

    :param shape: (longest, units)
    :param fit_mean: zero mean if False
    :param fit_var: unit var if False
    :param name:
    """
    units = shape[1]

    def instantiate(z):
        if fit_mean:
            # (B, M, dz)
            zero_mean = layers.Lambda(lambda t: K.zeros_like(t) + 0.001,
                                      output_shape=shape,
                                      name='{}.zero-mean'.format(name))(z)
            # (B, M, dz)
            prior_mean = layers.TimeDistributed(
                layers.Dense(units=units, use_bias=True, name='{}.transform-mean'.format(name)),
                name='{}.mean'.format(name))(zero_mean)
        else:
            # (B, M, dz)
            prior_mean = layers.Lambda(lambda t: K.zeros_like(t),
                                       output_shape=shape,
                                       name='{}.mean'.format(name))(z)
        if fit_var:
            # (B, M, dz)
            zero_log_var = layers.Lambda(lambda t: K.zeros_like(t) + 0.001,
                                         output_shape=shape,
                                         name='{}.unit-var'.format(name))(z)
            # (B, M, dz)
            prior_log_var = layers.TimeDistributed(
                layers.Dense(units=units, use_bias=True, name='{}.transform-var'.format(name)),
                name='{}.log-var'.format(name))(zero_log_var)
        else:
            # (B, M, dz)
            prior_log_var = layers.Lambda(lambda t: K.zeros_like(t),
                                          output_shape=shape,
                                          name='{}.log-var'.format(name))(z)
        return prior_mean, prior_log_var
    return instantiate


def KLStandardGaussian(shape, name='KL'):
    """
    KL from a parameterised isotropic Gaussian (diagonal covariance) to N(0, 1)

    Example: KLStandardGaussian(shape=(longest,))([mean, log_var])
    :param shape: (longest,)
    :param name:
    """
    def instantiate(args):
        mean, log_var = args
        return layers.Lambda(kl_from_q_to_standard_normal,
                             output_shape=shape,
                             name=name)([mean, log_var])
    return instantiate


def KLDiagonalGaussians(shape, name='KL'):
    """
    KL between two isotripic Gaussians.

    Example: KLDiagonalGaussians(shape=(longest,))([q_mean, q_log_var, p_mean, p_log_var])

    :param shape: (longest,)
    :param name:
    """
    def instantiate(args):
        mean, log_var, prior_mean, prior_log_var = args
        return layers.Lambda(kl_diagonal_gaussians,
                             output_shape=shape,
                             name=name)([mean, log_var, prior_mean, prior_log_var])
    return instantiate


def SelectSample(shape: tuple, deterministic_test=True, name='Select'):
    """
    Syntactic sugar for selecting deterministic output depending on training/testing mode.

    Returns a function that takes a pair (stochastic input, deterministic input).

    Example:
        - SelectSample(shape=(longest_x, dz), deterministic_test=True, 'Z')([z, mean])

    :param shape: output shape
    :param deterministic_test: whether we are testing deterministically
    :param name:
    """
    def instantiate(args):
        stochastic, deterministic = args
        if deterministic_test:
            return layers.Lambda(lambda d_s: K.in_test_phase(x=d_s[0], alt=d_s[1]),
                                 output_shape=shape,
                                 name=name)([deterministic, stochastic])
        else:
            return stochastic
    return instantiate
