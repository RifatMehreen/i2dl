"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    # Iterate through samples
    loss = 0.0
    dW = np.zeros_like(W)
    classes = range(0, W.shape[1], 1)

    for s_i, sample in enumerate(X):
        scores_per_class = np.zeros(len(classes))
        denom = 0.
        # Iterate through classes
        for class_ in classes:
            # Iterate through features/dimensions
            for f_i, feature in enumerate(sample):
                # Add score per sample per class
                scores_per_class[class_] += feature * W[f_i, class_]
                f_i += 1
        # For numerical stability, subtract max numerator from all numerators
        # Then take the log
        # Then add scores for later normalization
        max_score = max(scores_per_class)
        for sc_i, score in enumerate(scores_per_class):
            scores_per_class[sc_i] -= max_score
            scores_per_class[sc_i] = np.exp(score)
            denom += scores_per_class[sc_i]
        # Normalize scores
        for sc_i, _ in enumerate(scores_per_class):
            scores_per_class[sc_i] /= denom
        # Sum loss
        loss -= np.log(scores_per_class[y[s_i]])
        # Compute gradient
        for class_ in classes:
            for f_i, feature in enumerate(sample):
                if y[s_i] == class_:
                    dW[f_i, class_] += (scores_per_class[class_] - 1) * feature
                else:
                    dW[f_i, class_] += scores_per_class[class_] * feature
                f_i += 1
    # Divide loss by n
    loss /= s_i
    # Regularize loss and gradient
    for class_ in classes:
        for fi, _ in enumerate(dW):
            dW[fi, class_] /= s_i
            dW[fi, class_] += reg * W[fi, class_]
            loss += 0.5 * reg * W[fi, class_]**2
    ###########################################################################
    #                          END OF YOUR CODE                               #
    ###########################################################################
    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = y.shape[0]
    C = W.shape[1]

    exp = X@W - np.max(X@W)
    exp = np.exp(exp)
    sum_exp = np.sum(exp, axis=1)[:, None]
    scores = exp / sum_exp
    one_hot_targets = np.eye(C)[y.reshape(-1)]
    loss = -np.sum(np.log(np.sum(one_hot_targets * scores, axis=1))) / N
    loss += 0.5 * reg * np.sum(W * W)
    dW = X.T@(scores - one_hot_targets) / N + reg * W

    ##########################################################################
    #                          END OF YOUR CODE                               #
    ###########################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = np.linspace(5e-7, 8e-7, num=4)
    regularization_strengths = np.linspace(3e3, 5e3, num=5)

    for lr in learning_rates:
        train_acc = 0.
        val_acc = 0.
        for rs in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(
                X_train,
                y_train,
                learning_rate=lr,
                reg=rs,
                num_iters=1000)
            y_train_pred = softmax.predict(X_train)
            y_val_pred = softmax.predict(X_val)
            train_acc = np.mean(y_train == y_train_pred)
            val_acc = np.mean(y_val == y_val_pred)
            results[(lr, rs)] = (train_acc, val_acc)
            all_classifiers.append((softmax, val_acc.copy()))
            if val_acc > best_val:
                best_val = val_acc.copy()
                best_softmax = softmax
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
