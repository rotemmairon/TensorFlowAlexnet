from utils import *
from layers import *
import tensorflow as tf


class Alexnet(object):

    def __init__(self, data, keep_prob, num_outputs=17, reuse=False, verbose=False):
        self.verbose = verbose
        self.reuse = reuse
        self.output = self.build_model(data, num_outputs, keep_prob)

    def build_model(self, data, num_outputs, keep_prob):
        # Weights initialization function
        initializer = tf.uniform_unit_scaling_initializer()

        with tf.variable_scope('Conv2D1', reuse=self.reuse):
            conv = conv_layer("", data, [11, 11, 3, 96], [1, 4, 4, 1], initializer)
            conv = tf.nn.relu(conv)
            if self.verbose:
                print_activations(conv)

        with tf.name_scope('MaxPool2D'):
            pool1 = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            if self.verbose:
                print_activations(pool1)

        with tf.name_scope('LocalResponseNormalization'):
            lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)
            if self.verbose:
                print_activations(lrn1)

        with tf.variable_scope('Conv2D2', reuse=self.reuse):
            conv = conv_layer("", lrn1, [5, 5, 96, 256], [1, 1, 1, 1], initializer)
            conv = tf.nn.relu(conv)
            if self.verbose:
                print_activations(conv)

        with tf.name_scope('MaxPool2D'):
            pool2 = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            if self.verbose:
                print_activations(pool2)

        with tf.name_scope('LocalResponseNormalization'):
            lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)
            if self.verbose:
                print_activations(lrn2)

        with tf.variable_scope('Conv2D3', reuse=self.reuse):
            conv = conv_layer("", lrn2, [3, 3, 256, 384], [1, 1, 1, 1], initializer)
            conv = tf.nn.relu(conv)
            if self.verbose:
                print_activations(conv)

        with tf.variable_scope('Conv2D4', reuse=self.reuse):
            conv = conv_layer("", conv, [3, 3, 384, 384], [1, 1, 1, 1], initializer)
            conv = tf.nn.relu(conv)
            if self.verbose:
                print_activations(conv)

        with tf.variable_scope('Conv2D5', reuse=self.reuse):
            conv = conv_layer("", conv, [3, 3, 384, 256], [1, 1, 1, 1], initializer)
            conv = tf.nn.relu(conv)
            if self.verbose:
                print_activations(conv)

        with tf.name_scope('MaxPool2D'):
            pool5 = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            if self.verbose:
                print_activations(pool5)

        with tf.name_scope('LocalResponseNormalization'):
            lrn5 = tf.nn.local_response_normalization(pool5, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)
            if self.verbose:
                print_activations(lrn5)

        with tf.variable_scope('FullyConnected6', reuse=self.reuse):
            input_shape = lrn5.get_shape().as_list()
            n_inputs = int(np.prod(input_shape[1:]))
            inference = tf.reshape(lrn5, [-1, n_inputs])
            fc6 = fc_layer("", inference, [n_inputs, 4096])
            fc6 = tf.tanh(fc6)
            if self.verbose:
                print_activations(fc6)

        with tf.name_scope('Dropout'):
            drop6 = tf.nn.dropout(fc6, keep_prob, name='drop6')
            if self.verbose:
                print_activations(drop6)

        with tf.variable_scope('FullyConnected7', reuse=self.reuse):
            fc7 = fc_layer("", drop6, [4096, 4096])
            fc7 = tf.tanh(fc7)
            if self.verbose:
                print_activations(fc7)

        with tf.name_scope('Dropout'):
            drop7 = tf.nn.dropout(fc7, keep_prob, name='drop7')
            if self.verbose:
                print_activations(drop7)

        with tf.variable_scope('FullyConnected8', reuse=self.reuse):
            fc8 = fc_layer("", drop7, [4096, num_outputs])
            if self.verbose:
                print_activations(fc8)

        return fc8
