import tensorflow as tf
import numpy as np


def layernorm(x):
    return tf.contrib.layers.layer_norm(x)


def dropout(x, training, rate=.3):
    return tf.layers.dropout(x, training=training, rate=rate)


def leaky(x):
    return tf.maximum(x, .2*x)


def dense(x, units):
    return tf.layers.dense(x, units)


def get_tf_dataset_pipeline(X, y, batch_size, shuffle=True):
    """
    Creates a tensorflow dataset input pipeline from a split of data

    Parameters
    ----------
    X: pandas dataframe of features
    y: pandas dataframe of labels
    (These should be aligned for concatenation)

    batch_size: size of each minibatch

    Return
    ----------
    A tensorflow dataset iterator that provides one batch with the
    label as the first element in each sample
    """

    X = X.values
    y = np.expand_dims(y.values, axis=-1)
    full = tf.concat([y, X], axis=-1)
    if shuffle:
        dataset = tf.data.Dataset.from_tensor_slices(full) \
            .shuffle(10000).repeat().batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(full) \
            .repeat().batch(batch_size)
    dataset = dataset.prefetch(1).make_one_shot_iterator().get_next()
    return dataset
