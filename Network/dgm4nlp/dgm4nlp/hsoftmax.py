import numpy as np
from theano.tensor.nnet import h_softmax, softmax
from keras import backend as K
from keras.engine import Layer
from keras import initializers
from keras import layers


class TimeDistributedHierarchicalSoftmax(Layer):
    """Two-layer Hierarchical Softmax layer. Provides an approximate
    softmax that is much faster to compute in cases where there are a
    large number (~10K+) of classes.
    # Input shape
        A list of two tensors:
           - The first tensor should have shape (nb_samples, dim) and represents the input feature vector
           - The second tensor should have shape (nb_samples,), have integer type, and represent the
           labels of each training example.
    # Output shape
        1D Tensor with shape (nb_samples,) representing the negative log probability of the correct class
    # Arguments
        total_outputs: How many outputs the hierarchical softmax is over
        per_class: How many outputs per top level class (should be on the order of the square root of the total number of classes)
    # References
    - [Classes for Fast Maximum Entropy Training](http://arxiv.org/pdf/cs/0108006.pdf)
    - [Hierarchical Probabilistic Neural Network Language Model](http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf)
    - [Strategies for Training Large Vocabulary Neural Language Models](http://arxiv.org/pdf/1512.04906)
    """

    def __init__(self, total_outputs, per_class=None,
                 top_weights_init='glorot_uniform', top_bias_init='zero',
                 bottom_weights_init='glorot_uniform', bottom_bias_init='zero',
                 **kwargs):
        assert K.backend() == 'theano', "HierarchicalSoftmax only supported by Theano"

        if per_class is None:
            per_class = int(np.ceil(np.sqrt(total_outputs)))

        self.total_outputs = total_outputs
        self.per_class = per_class

        self.n_classes = int(np.ceil(self.total_outputs * 1. / self.per_class))
        self.n_outputs_actual = self.n_classes * self.per_class

        self.top_weights_init = initializers.get(top_weights_init)
        self.top_bias_init = initializers.get(top_bias_init)

        self.bottom_weights_init = initializers.get(bottom_weights_init)
        self.bottom_bias_init = initializers.get(bottom_bias_init)

        assert self.n_outputs_actual >= self.total_outputs, \
            "The number of actual HSM outputs must be at least the number of outputs you're modeling over."
        super(TimeDistributedHierarchicalSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        print('input_shape', input_shape, 'input_dim', input_dim)

        self.top_weights = self.add_weight(shape=(input_dim, self.n_classes,),
                                           initializer=self.top_weights_init,
                                           trainable=True,
                                           name='{}_top_weights'.format(self.name))

        self.top_bias = self.add_weight(shape=(self.n_classes,),
                                        initializer=self.top_bias_init,
                                        trainable=True,
                                        name='{}_top_bias'.format(self.name))

        self.bottom_weights = self.add_weight(shape=(self.n_classes, input_dim, self.per_class,),
                                              initializer=self.bottom_weights_init,
                                              trainable=True,
                                              name='{}_bottom_weights'.format(self.name))

        self.bottom_bias = self.add_weight(shape=(self.n_classes, self.per_class,),
                                           initializer=self.bottom_bias_init,
                                           trainable=True,
                                           name='{}_bottom_bias'.format(self.name))
        super(TimeDistributedHierarchicalSoftmax, self).build(input_shape)

    def call(self, inputs):
        if type(inputs) is list:
            batch, target = inputs
            t_reshaped = K.flatten(target)
        else:
            batch = inputs
            t_reshaped = None
        b_reshaped = K.reshape(batch, (batch.shape[0] * batch.shape[1], batch.shape[2]))
        px = h_softmax(b_reshaped, b_reshaped.shape[0],
                       self.total_outputs, self.n_classes, self.per_class,
                       self.top_weights, self.top_bias,
                       self.bottom_weights, self.bottom_bias,
                       target=t_reshaped)
        if t_reshaped is None:
            # (B, M)
            return K.reshape(px, (batch.shape[0], batch.shape[1]))
        else:
            # (B, M, V)
            return K.reshape(px, (batch.shape[0], batch.shape[1], -1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.total_outputs

    def get_config(self):
        config = {'total_output': self.total_outputs,
                  'per_class': self.per_class,
                  'n_classes': self.n_classes,
                  'n_outputs_actual': self.n_outputs_actual,
                  'top_weights_init': self.top_weights_init,
                  'top_bias_init': self.top_bias_init,
                  'bottom_weights_init': self.bottom_weights_init,
                  'bottom_bias_init': self.bottom_bias_init}
        base_config = super(TimeDistributedHierarchicalSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def main():
    longest = 10
    dx = 12
    vocab = 100
    # (B, M)
    x = layers.Input((longest, ), dtype='int64')
    # (B, M, dx)
    emb = layers.Embedding(input_dim=vocab, output_dim=dx, input_length=longest)(x)
    px = TimeDistributedHierarchicalSoftmax(vocab)([emb, x])

    from keras.models import Model
    model = Model(inputs=x, outputs=[emb, px])

    import numpy as np
    X = np.random.randint(0, vocab, size=(2, longest))
    print(X)
    E, PX = model.predict(X)
    print('Embeddings')
    print(E)
    print(E.shape)
    print('PX')
    print(PX)
    print(PX.shape)

if __name__ == '__main__':
    main()