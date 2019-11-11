import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w, step=1):
    return tf.nn.conv2d(x, w, strides=[1, step, step, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dense(inputs, input_size, output_size, activation_function=None, keep_prob=1.0, return_param=False):
    w = tf.Variable(xavier_init(input_size, output_size))
    b = tf.Variable(tf.zeros([1, output_size]))
    Wx_plus_b = tf.matmul(inputs, w) + b
    if activation_function is None:
        outputs = tf.nn.dropout(Wx_plus_b, keep_prob)
    else:
        outputs = activation_function(tf.nn.dropout(Wx_plus_b, keep_prob))
    if return_param:
        return outputs, w, b
    else:
        return outputs


def cnn_dense(inputs, dim0, dim1, in_channels, out_channels, activation_function=None):
    w = weight_variable([5, 5, in_channels, out_channels])
    b = bias_variable([out_channels])
    if activation_function is None:
        h_conv = conv2d(inputs, w, 2) + b
    else:
        h_conv = activation_function(conv2d(inputs, w, 2) + b)
    dim0, dim1 = (dim0 + 1) // 2, (dim1 + 1) // 2
    h_pool = max_pool_2x2(h_conv)
    dim0, dim1 = (dim0 + 1) // 2, (dim1 + 1) // 2
    return h_pool, dim0, dim1


def add_noise(layer, C, eps):
    layer_clip = layer / tf.maximum(1., tf.reduce_max(tf.abs(layer)) / C)
    layer_with_noise = tfp.distributions.Laplace(layer_clip, C / eps).sample()
    return layer_with_noise


if __name__ == '__main__':
    pass
