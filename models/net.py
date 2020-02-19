import tensorflow as tf
from models.nn_utils import dense, leaky


def network(x, layers, training):
    """
    Create a regression network with only one hidden layer.

    Parameters
    ----------
    features: input to the network
    units: list of number of hidden units in the fully-connected layer

    Return
    ----------
    x: the predicted output given features
    """

    with tf.variable_scope('regressor', reuse=tf.AUTO_REUSE):
        for units in layers:
            x = dense(x, units)
            x = leaky(x)
        x = dense(x, 1)
        return x
