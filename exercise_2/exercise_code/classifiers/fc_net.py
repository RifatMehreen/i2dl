import numpy as np

from exercise_code.layers import *
from exercise_code.layer_utils import *


def normal_generator(mean, std, size):
    dist = np.random.normal(mean, std, size)
    dist = (dist - np.mean(dist)) * (std / np.std(dist)) + mean
    return dist


def softmax(x):
    exp = x - np.max(x)
    exp = np.exp(exp)
    sum_exp = np.sum(exp, axis=1)[:, None]
    scores = exp / sum_exp
    return scores


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        W1 = normal_generator(
            mean=0.0, std=weight_scale, size=input_dim * hidden_dim)
        W1 = W1.reshape((input_dim, hidden_dim))
        b1 = np.zeros(shape=hidden_dim)
        W2 = normal_generator(
            mean=0.0, std=weight_scale, size=hidden_dim * num_classes)
        W2 = W2.reshape((hidden_dim, num_classes))
        b2 = np.zeros(shape=num_classes)
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        scores, cache_1 = affine_relu_forward(X, W1, b1)
        scores, cache_2 = affine_forward(scores, W2, b2)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        data_loss, dx = softmax_loss(scores, y)
        reg_loss = self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + 0.5 * reg_loss
        dx2, dw2, db2 = affine_backward(dx, cache_2)
        dw2 += self.reg * W2
        dx1, dw1, db1 = affine_relu_backward(dx2, cache_1)
        dw1 += self.reg * W1

        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw2
        grads['b2'] = db2

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        i = 1
        prev_dim = input_dim
        all_dims = hidden_dims
        all_dims.append(num_classes)
        # Init weights
        for dim in all_dims:
            W = normal_generator(
                mean=0.0, std=weight_scale, size=prev_dim * dim)
            W = W.reshape((prev_dim, dim))
            self.params['W' + str(i)] = W
            self.params['b' + str(i)] = np.zeros(shape=dim)
            if self.use_batchnorm:
                self.params['gamma' + str(i)] = 1.
                self.params['beta' + str(i)] = 0.
            i += 1
            prev_dim = dim
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        caches = {}
        # Perfrom first L-1 forward passes
        for i in range(1, self.num_layers):
            w = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            # First layer
            if i == 1:
                scores, a_cache = affine_forward(X, w, b)
            # Other layers
            else:
                scores, a_cache = affine_forward(scores, w, b)
            # Save cache for affine layer
            caches['affine' + str(i)] = a_cache
            # Batchnorm layer
            if self.use_batchnorm:
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                scores, b_cache = batchnorm_forward(
                    scores, gamma, beta, self.bn_params[i - 1])
                caches['batch' + str(i)] = b_cache
            # ReLu layer
            scores, r_cache = relu_forward(scores)
            caches['relu' + str(i)] = r_cache
            # Dropout layer
            if self.use_dropout:
                scores, d_cache = dropout_forward(scores, self.dropout_param)
                caches['dropout' + str(i)] = d_cache
        # Last layer
        last_idx = self.num_layers
        w = self.params['W' + str(last_idx)]
        b = self.params['b' + str(last_idx)]
        scores, f_cache = affine_forward(scores, w, b)
        caches['affine' + str(last_idx)] = f_cache
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        # Loss and backward pass for last layer
        data_loss, dout = softmax_loss(scores, y)
        reg_loss = 0.
        final_cache = caches['affine' + str(last_idx)]
        dout, dw, db = affine_backward(dout, final_cache)
        grads['W' + str(self.num_layers)] = dw + self.reg * w
        grads['b' + str(self.num_layers)] = db
        # Backward pass for first L-1 layers
        for i in range(self.num_layers - 1, 0, -1):
            # Dropout backward
            if self.use_dropout:
                d_cache = caches['dropout' + str(i)]
                dout = dropout_backward(dout, d_cache)
            # Batch dropout
            r_cache = caches['relu' + str(i)]
            a_cache = caches['affine' + str(i)]
            if self.use_batchnorm:
                # ReLU dropout
                dout = relu_backward(dout, r_cache)
                b_cache = caches['batch' + str(i)]
                dout, dgamma, dbeta = batchnorm_backward(dout, b_cache)
                grads['gamma' + str(i)] = dgamma
                grad['beta' + str(i)] = dbeta
                # Affine dropout
                dout, dw_i, db_i = affine_backward(dout, a_cache)
            # Affine + relu
            else:
                dout, dw_i, db_i = affine_relu_backward(
                    dout, (a_cache, r_cache))
            # Save to grad dict
            # TODO: WTF do I need to remove reg for this to work?!
            w_i = self.params['W' + str(i)]
            grads['W' + str(i)] = dw_i  + self.reg * w_i
            grads['b' + str(i)] = db_i
            # Reg loss
        # TODO : WTF does calculating reg_loss destroy the results?!?
        #reg_loss += np.sum(w * w)
        loss = data_loss + 0.5 * reg_loss
        return loss, grads
