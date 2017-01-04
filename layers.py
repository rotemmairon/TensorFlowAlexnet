import tensorflow as tf


def conv_layer(name, incoming, shape, stride, initializer):
    kernel = tf.get_variable(name + 'W', shape=shape, initializer=initializer)
    conv = tf.nn.conv2d(incoming, kernel, stride, padding='SAME')
    biases = tf.get_variable(name + 'b', shape=shape[-1:], initializer=tf.constant_initializer(0.0))
    add = tf.nn.bias_add(conv, biases)
    return add


def fc_layer(name, incoming, shape):
    weights = tf.get_variable(name + 'W', shape=shape,
                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02))
    biases = tf.get_variable(name + 'b', shape=shape[1], initializer=tf.constant_initializer(0.0))
    matmul = tf.matmul(incoming, weights)
    add = tf.nn.bias_add(matmul, biases)
    return add
