from __future__ import division

import numpy as np
import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


# tensorflow/magenta/blob/master/magenta/models/nsynth/wavenet/masked.py#L53
def time_to_batch(x, dilation):
    with tf.name_scope('time_to_batch'):
        shape = x.get_shape().as_list()
        y = tf.reshape(x, [
            shape[0], shape[1] // dilation, dilation, shape[2]
        ])
        y = tf.transpose(y, [0, 2, 1, 3])
        y = tf.reshape(y, [
            shape[0] * dilation, shape[1] // dilation, shape[2]
        ])
        y.set_shape([
            mul_or_none(shape[0], dilation),
            mul_or_none(shape[1], 1. / dilation),
            shape[2]
        ])
        return y


# tensorflow/magenta/blob/master/magenta/models/nsynth/wavenet/masked.py#L85
def batch_to_time(x, dilation):
    with tf.name_scope('batch_to_time'):
        shape = x.get_shape().as_list()
        y = tf.reshape(x, [shape[0] // dilation, dilation, shape[1], shape[2]])
        y = tf.transpose(y, [0, 2, 1, 3])
        y = tf.reshape(y, [shape[0] // dilation, shape[1] * dilation, shape[2]])
        y.set_shape([mul_or_none(shape[0], 1. / dilation),
                     mul_or_none(shape[1], dilation),
                     shape[2]])
        return y


# tensorflow/magenta/blob/master/magenta/models/nsynth/wavenet/masked.py#L106
def conv1d(x, filt, biases, dilation=1, causal=True):
    """Fast 1D convolution that supports causal padding and dilation.
    
      :param x: the [bs, time, input_channes] float input tensor
      :param filt: the filter we convolve x with
      :param biases: the biases
      :param dilation: the amount of dilation
      :param causal: whether it's a causal convolution or not
      :return: the 1D convolution output
    """
    batch_size, length, num_input_channels = x.get_shape().as_list()
    filter_width, filt_in_channels, filt_out_channels = filt.get_shape().as_list()
    assert length % dilation == 0
    assert num_input_channels == filt_in_channels

    padding = 'VALID' if causal else 'SAME'

    x_ttb = time_to_batch(x, dilation)
    if filter_width > 1 and causal:
        x_ttb = tf.pad(x_ttb, [[0, 0], [filter_width - 1, 0], [0, 0]])

    y = tf.nn.conv1d(x_ttb, filt, 1, padding)
    y = tf.nn.bias_add(y, biases)
    y = batch_to_time(y, dilation)
    y.set_shape([batch_size, length, filt_out_channels])
    return y


# tensorflow/magenta/blob/master/magenta/models/nsynth/utils.py#L64
def mu_law_encode(audio, quantization_channels=256):
    """Quantizes waveform amplitudes."""
    with tf.name_scope('encode'):
        mu = quantization_channels - 1
        out = tf.sign(audio) * tf.log(1 + mu * tf.abs(audio)) / np.log(1 + mu)
        out = tf.cast(tf.floor(out * 128), tf.int8)
        return out


# tensorflow/magenta/blob/master/magenta/models/nsynth/utils.py#L79
def mu_law_decode(x, quantization_channels=256):
    """Recovers waveform from quantized values."""
    with tf.name_scope('decode'):
        x = tf.cast(x, tf.float32)
        mu = quantization_channels - 1
        out = (x + 0.5) * 2. / (mu + 1)
        out = tf.sign(out) / mu * ((1 + mu) ** tf.abs(out) - 1)
        out = tf.where(tf.equal(x, 0), x, out)
        return out


# tensorflow/magenta/blob/master/magenta/models/nsynth/wavenet/masked.py#L36
def mul_or_none(a, b):
    """Return the element wise multiplicative of the inputs.
  
    If either input is None, we return None.
  
    Args:
      a: A tensor input.
      b: Another tensor input with the same type as a.
  
    Returns:
      None if either input is None. Otherwise returns a * b.
    """
    if a is None or b is None:
        return None
    return a * b


# tensorflow/magenta/blob/master/magenta/models/nsynth/wavenet/masked.py#L20
def shift_right(x):
    """Shift the input over by one and a zero to the front.
  
    Args:
      x: The [mb, time, channels] tensor input.
  
    Returns:
      x_sliced: The [mb, time, channels] tensor output.
    """
    shape = x.get_shape().as_list()
    x_padded = tf.pad(x, [[0, 0], [1, 0], [0, 0]])
    x_sliced = tf.slice(x_padded, [0, 0, 0], tf.stack([-1, shape[1], -1]))
    x_sliced.set_shape(shape)
    return x_sliced
