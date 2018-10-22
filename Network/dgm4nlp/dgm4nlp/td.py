"""
Syntactic sugar for time distributed (td) layers.

:Authors: - Wilker Aziz
"""
from keras import layers
from dgm4nlp.hsoftmax import TimeDistributedHierarchicalSoftmax


def MLP(specs, name='MLP'):
    """
    Applies an MLP to every time step in a sequence.

    Example: MLP([(50, 'tanh'), (50, 'relu'), (20, 'linear')])(emb)
        where emb.shape is (B, M, d)

    :param specs: configure hidden layers (use pairs (output dimension, activation function)
        e.g. [(50, 'tanh'), (50, 'relu')]
        e.g. [(50, 'tanh')] * 5
    :param name: prefix for name of layers
    """
    def instantiate(inputs):
        if not specs:
            return inputs
        else:
            units, activation = specs[0]
            h = layers.TimeDistributed(layers.Dense(units,
                                                    activation=activation,
                                                    name='{}.{}.Dense'.format(name, len(specs) - 1)),
                                       name='{}.{}'.format(name, len(specs) - 1))(inputs)
            return MLP(specs[1:], name=name)(h)
    return instantiate


def Activation(units, activation, use_bias=True, hidden_layers=[], name='Activation'):
    """
    Syntactic sugar for a time distributed activation layer.

    Examples:
        - Activation(dz, 'relu', use_bias=True)(zeros)
        - Activation(dz, 'softplus', use_bias=True)(ones)
        - Activation(dz, 'linear', use_bias=False)(ones)
        - Activation(dz, 'softmax', use_bias=True)(ones)

    :param units: number of output units
    :param activation: activation function
    :param use_bias: whether or not to use bias.
    :param hidden_layers: optionally pass input through an MLP first
        specify layers as list of pairs (number of units, activation function)
    :param name:
    """
    def instantiate(inputs):
        if hidden_layers:  # transform inputs through an MLP
            inputs = MLP(specs=hidden_layers, name='{}.MLP'.format(name))(inputs)
        return layers.TimeDistributed(layers.Dense(units,
                                                   activation=activation,
                                                   use_bias=use_bias,
                                                   name='{}.Activation'.format(name)),
                                      name=name)(inputs)
    return instantiate


def Softmax(units, use_bias=True, hidden_layers=[], name='Softmax'):
    """
    Syntactic sugar for a time distributed softmax.

    Example:
        - Softmax(units=vocab_size)(h)  # computes softmax(Wh + b) which is same as ld.Activation(vocab_size, 'softmax')
        - Softmax(units=vocab_size, use_bias=False)(h)  # computes softmax(Wh)

    :param units: size of the support
    :param use_bias: whether or not to use bias
    :param hidden_layers: optionally pass input through an MLP first
        specify layers as list of pairs (number of units, activation function)
    :param name:
    """
    return Activation(units=units, activation='softmax', use_bias=use_bias, hidden_layers=hidden_layers, name=name)


def HierarchicalSoftmax(units, hidden_layers=[], name='HierarchicalSoftmax'):
    """
    Syntactic sugar for a time distributed hierarchical softmax.

    Example:
        - HierarchicalSoftmax(vocab_size)([embeddings, labels])

    :param units: size of the support
    :param hidden_layers: optionally pass input through an MLP first
        specify layers as list of pairs (number of units, activation function)
    :param name:
    """
    def instantiate(inputs):
        if hidden_layers:  # transform inputs through an MLP
            inputs = MLP(specs=hidden_layers, name='{}.MLP'.format(name))(inputs)
        return TimeDistributedHierarchicalSoftmax(units, name=name)(inputs)
    return instantiate


def Linear(units, use_bias=True, hidden_layers=[], name='Linear'):
    """
    Syntactic sugar for a time distributed linear layer.

    Examples:
        - Linear(dz, use_bias=True)(zeros)  # same as ld.Activation(dz, 'linear')
        - Linear(dz, use_bias=False)(ones)

    :param units: number of output units
    :param use_bias: whether or not to use bias.
    :param hidden_layers: optionally pass input through an MLP first
        specify layers as list of pairs (number of units, activation function)
    :param name:
    """
    return Activation(units=units, activation='linear', use_bias=use_bias, hidden_layers=hidden_layers, name=name)


def RNN(units, dropout=0., recurrent_dropout=0., nb_layers=1,
        architecture='lstm', name='RNN'):
    """
    Bidirectional recurrent layers.

    Example:
        BiRNN(100)(embeddings)

    :param units: number of LSTM units
    :param dropout:
    :param recurrent_dropout:
    :param merge_mode: how to merge fwd/bwd LSTMs
    :param nb_layers: number of layers (defaults to 1), 0 is supported and obviously does nothing
    :param architecture: one of 'lstm' or 'gru' (defaults to 'lstm')
    :param name:
    """

    def instantiate(inputs):
        if architecture is 'gru':
            RNN = layers.GRU
        else:
            RNN = layers.LSTM
        for i in range(nb_layers):
            inputs = RNN(units=units,
                         return_sequences=True,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         name='{}.{}-{}'.format(name, architecture, i))(inputs)
        return inputs
    return instantiate


def BiRNN(units, dropout=0., recurrent_dropout=0., merge_mode='concat', nb_layers=1,
          architecture='lstm', name='BiRNN'):
    """
    Bidirectional recurrent layers.

    Example:
        BiRNN(100)(embeddings)

    :param units: number of LSTM units
    :param dropout:
    :param recurrent_dropout:
    :param merge_mode: how to merge fwd/bwd LSTMs
    :param nb_layers: number of layers (defaults to 1), 0 is supported and obviously does nothing
    :param architecture: one of 'lstm' or 'gru' (defaults to 'lstm')
    :param name:
    """
    def instantiate(inputs):
        if architecture is 'gru':
            RNN = layers.GRU
        else:
            RNN = layers.LSTM
        for i in range(nb_layers):
            inputs = layers.Bidirectional(RNN(units=units,
                                              return_sequences=True,
                                              dropout=dropout,
                                              recurrent_dropout=recurrent_dropout,
                                              name='{}.{}-{}'.format(name, architecture, i)),
                                          merge_mode=merge_mode,
                                          name='{}.Bi{}-{}'.format(name, architecture, i))(inputs)
        return inputs
    return instantiate