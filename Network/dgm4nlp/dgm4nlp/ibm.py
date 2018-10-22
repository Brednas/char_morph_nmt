"""
Building blocks for IBM models.

Functions are constructors for layers, that is, they return a function that instantiates layers.

Classes are used to wrap around these functions normalising their use.

:Authors: - Wilker Aziz
"""
from keras import backend as K
from keras import layers
from dgm4nlp import blocks
from dgm4nlp import td


def LengthDistribution(longest, name='LengthDistribution'):
    def instantiate(x):
        mask = blocks.MakeMask(shape=(longest,), dtype=K.floatx(),
                               name='{}.Mask'.format(name))(x)
        # Length distribution P(n|m) = 1/n
        # (B, longest)
        return layers.Lambda(lambda t: t / K.sum(t, axis=1)[:, None],  # normalise mask by total length
                             output_shape=(longest,),
                             name=name)(mask)
    return instantiate


def UniformAlignment(longest_x, longest_y, name='UniformAlignment'):
    """
    Uniform distribution over x's tokens.

    Example: UniformAlignent(longest_x, longest_y)(x)

    :param longest_x: longest possible x-sequence (M)
    :param longest_y: longest possible y-sequence (N)
    :param name:
    :return: a function that takes x and produces P(A|X,N)
        where x is (B, M) and pa is (B, N, M)
    """

    def uniform_weights(x_mask, y_mask):  # (B, M), (B, N)
        # normalise x_mask by total x_length and broadcast N
        return (x_mask / K.sum(x_mask, axis=1)[:, None])[:, None, :] + K.zeros_like(y_mask)[:, :, None]

    def instantiate(args):
        # Get a mask from x in order to compute a uniform distribution over its tokens
        x, y = args
        # (B, M)
        x_mask = blocks.MakeMask(shape=(longest_x,), dtype=K.floatx(),
                                     name='{}.X-Mask'.format(name))(x)
        # (B, N)
        y_mask = blocks.MakeMask(shape=(longest_y,), dtype=K.floatx(),
                                 name='{}.Y-Mask'.format(name))(y)
        # Uniform distribution
        # (B, N, M)
        pa_x = layers.Lambda(lambda pair: uniform_weights(pair[0], pair[1]),
                             output_shape=(longest_x, longest_y),
                             name=name)([x_mask, y_mask])
        return pa_x
    return instantiate


def PositionCPD(longest_x, longest_y,
                dm, dn, dj,
                context_layers=[],
                max_m=None, max_n=None, max_j=None,
                dynamic_support=False, name='PositionCPD'):
    """
    A distribution over positions i in x given (m, n, j).

    Example: PositionCPD(longest_x, longest_y, dm=10, dn=10, dj=10, context_layers=[10, 'relu'])

    :param longest_x: longest x-sequence (M)
    :param longest_y: longest y-sequence (N)
    :param dm: m's embedding size
    :param dn: n's embedding size
    :param dj: j's embedding size
    :param context_layers: specification of layers to encode the context
        - the context is a tuple (m, n, j) per observation where m, n and j are embedded
        - specify layers by specifying pairs (number of units, activation function)
    :param max_m: maximum m for clipping (defaults to longest_x + 1)
    :param max_n: maximum n for clipping (defaults to longest_y + 1)
    :param max_j: maximum j for clipping (defaults to longest_y + 1)
    :param dynamic_support: whether the support of P(A|m,n,j) is fixed (longext_x) or dynamic (m)
        dynamic support is implemented by masking i > m before normalisation (instead of after normalisation)
    :param name:
    :return: a function that takes [x, y] and produces P(A|X,N)
        where x is (B, M), y is (B, N) and pa is (B, N, M)
    """

    if max_m is None:
        max_m = longest_x + 1
    if max_n is None:
        max_n = longest_y + 1
    if max_j is None:
        max_j = longest_y + 1

    def instantiate(args):
        x, y = args

        # 1. First we figure out M, N, and J
        # (B, M)
        x_mask = blocks.MakeMask(shape=(longest_x,), dtype='int64',
                                 name='{}.X-Mask'.format(name))(x)

        # Clipped length
        # (B,)  -- length is integer because it will be embedded
        x_length = layers.Lambda(lambda t: K.sum(t, axis=-1),
                                 output_shape=(1,),
                                 name='{}.X-Length'.format(name))(x_mask)

        # (B, N)
        y_mask = blocks.MakeMask(shape=(longest_y,), dtype='int64',
                                 name='{}.Y-Mask'.format(name))(y)
        # (B,)  -- length is integer because it will be embedded
        y_length = layers.Lambda(lambda t: K.sum(t, axis=-1),
                                 output_shape=(1,),
                                 name='{}.Y-Length'.format(name))(y_mask)

        # Copy x's length (m) into every position of y's mask
        # (B, N)
        m = layers.Lambda(lambda pair: pair[0] * pair[1][:, None], output_shape=(longest_y,))([y_mask, x_length])
        # Embed m
        m = layers.Embedding(input_dim=longest_x + 1, output_dim=dm, input_length=longest_y)(m)

        # (B, N)
        n = layers.Lambda(lambda pair: pair[0] * pair[1][:, None], output_shape=(longest_y,))([y_mask, y_length])
        # Embed n
        n = layers.Embedding(input_dim=longest_y + 1, output_dim=dn, input_length=longest_y)(n)

        # Clipped positions

        # (B, N) -- position is integer because it will be embedded
        j = layers.Lambda(lambda t: K.zeros_like(t, dtype='int64') + K.arange(1, longest_y + 1, dtype='int64'),
                          output_shape=(longest_y,),
                          name='{}.Y-Positions'.format(name))(y)
        # (B, N, dj)
        j = layers.Embedding(input_dim=longest_y + 1,
                             output_dim=dj,
                             input_length=longest_y,
                             name='{}.J'.format(name))(j)

        # 3. Then we make a context for the alignment distribution

        # Note: Keras's Concatenate layer breaks TimeDistributed, thus I use K.concatenate
        # (B, N, dm + dn + dj)
        c = layers.Lambda(lambda triple: K.concatenate(triple, axis=-1),
                          output_shape=(longest_y, dm + dn + dj),
                          name='{}.M-N-J'.format(name))([m, n, j])

        if dynamic_support:
            # (B, N, M)
            u = td.Linear(longest_x, hidden_layers=context_layers, name='{}.LogPotentials'.format(name))(c)
            # Exponentiate and mask
            # (B, N, M)
            u = layers.Lambda(lambda pair: K.exp(pair[0]) * pair[1][:, None, :],
                              output_shape=(longest_y, longest_x),
                              name='{}.Potentials'.format(name))([u, x_mask])
            # Normalise
            # (B, N, M)
            return layers.Lambda(lambda t: t / (K.sum(t, axis=-1)[:, :, None]),
                                 output_shape=(longest_y, longest_x),
                                 name=name)(u)
        else:
            # Compute a distribution over the maximum number of x-positions
            # NOTE: in principle this Softmax should be defined over max_m position,
            #  but for now I do not want to bother converting the resulting matrix to (B, N, M)
            #  thus I am using longest_x instead.
            #  I need the final matrix to be (B, N, M) because that's the shape the rest of the model expects
            # (B, N, M)
            pa = td.Softmax(longest_x, hidden_layers=context_layers, name='{}.LogPotentials'.format(name))(c)
            # Mask invalid input positions
            # (B, N, M)
            return blocks.ApplyMask(shape=(longest_y, longest_x), broadcast_axis=1, name=name)([pa, x_mask])
    return instantiate


class AlignmentComponent:
    """
    An alignment component to be used as a building block in a model.
    This class is used to normalise the use (configuration, etc.) of different architectures for
    distributions over directional alignments.

    The constructor of the component configures the component (architecture details).
    Calling the component returns a function to construct and instantiate the architecture.

    To get a constructor for an alignment component one must pass
        longest_x, longest_y and a name

    actually constructing the architecture requires
        x, y

    Example:

        component = ExampleComponent(units=10, activation='relu')
        constructor = component(longest_x, longest_y, 'pa_x')
        pa_x = constructor(x, y)
    """

    def __init__(self):  # here you configure your alignment components
        pass

    def __call__(self, longest_x, longest_y, name):  # here you construct the layers that implement the component
        """
        Returns a function that takes [x, y] and produces P(A|X,N)
            where x is (B, M), y is (B, N) and pa is (B, N, M)
        :param longest_x: M
        :param longest_y:  N
        :param name:
        :return: P(A|X,N) with shape (B, N, M)
        """
        pass


class UniformAlignmentComponent(AlignmentComponent):
    """
    Wrapper for architecture: UniformAlignment.
    """

    def __init__(self):  # nothing to configure
        pass

    def __call__(self, longest_x, longest_y, name):
        return UniformAlignment(longest_x, longest_y, name=name)

    def __repr__(self):
        return '%s.%s()' % (UniformAlignmentComponent.__module__, UniformAlignmentComponent.__name__)


class PositionCPDComponent(AlignmentComponent):
    """
    Wrapper for architecture: PositionCPD.
    """

    def __init__(self, dm, dn, dj, context_layers=[],
                 max_m=None, max_n=None, max_j=None,
                 dynamic_support=False):
        self.dm = dm
        self.dn = dn
        self.dj = dj
        self.context_layers = context_layers
        self.max_m = max_m
        self.max_n = max_n
        self.max_j = max_j
        self.dynamic_support = dynamic_support

    def __call__(self, longest_x, longest_y, name):
        return PositionCPD(longest_x, longest_y, name=name,
                           dm=self.dm, dn=self.dn, dj=self.dj,
                           context_layers=self.context_layers,
                           max_m=self.max_m,
                           max_n=self.max_n,
                           max_j=self.max_j,
                           dynamic_support=self.dynamic_support)

    def __repr__(self):
        return '%s.%s(dm=%r, dn=%r, dj=%r, context_layers=%r, max_m=%r, max_n=%r, max_j=%r, dynamic_support=%r)' % (
            PositionCPDComponent.__module__, PositionCPDComponent.__name__,
            self.dm, self.dn. self.dj, self.context_layers, self.max_m, self.max_n, self.max_j, self.dynamic_support)

