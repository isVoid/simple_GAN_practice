import tensorflow as tf
import numpy as np

global USE_TENSORBOARD

def lrelu(x):
    return tf.maximum(x, 0.2*x)

def bnorm(x):
    return tf.nn.batch_normalization(x,
                                    mean=0,
                                    variance = 1,
                                    offset = 0,
                                    scale = 1,
                                    variance_epsilon = 1e-5)

def conv_lrelu(input_, in_channel, out_channel, name="conv", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [5, 5, in_channel, out_channel], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        conv = tf.nn.conv2d(input_, w, strides = [1, 1, 1, 1], padding = "SAME")
        act = lrelu(conv + b)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def conv_bn_lrelu(input_, in_channel, out_channel, name="conv", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [5, 5, in_channel, out_channel], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        conv = tf.nn.conv2d(input_, w, strides = [1, 1, 1, 1], padding = "SAME")
        bn = bnorm(conv + b)
        act = lrelu(bn)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def conv_relu(input_, in_channel, out_channel, name="conv", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [5, 5, in_channel, out_channel], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        conv = tf.nn.conv2d(input_, w, strides = [1, 1, 1, 1], padding = "SAME")
        act = tf.nn.relu(conv + b)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def conv_sigmoid(input_, in_channel, out_channel, name="conv", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [5, 5, in_channel, out_channel], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        conv = tf.nn.conv2d(input_, w, strides = [1, 1, 1, 1], padding = "SAME")
        act = tf.nn.sigmoid(conv + b)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def conv_tran_bn_relu(input_, output_shape, name="conv_t", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [4, 4, output_shape[-1], input_.shape[-1]], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        conv_t = tf.nn.conv2d_transpose(input_, w, output_shape, strides=[1, 2, 2, 1], padding = "SAME")
        bn = bnorm(conv_t + b)
        act = tf.nn.relu(bn)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def conv_tran_bn_lrelu(input_, output_shape, name="conv_t", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [4, 4, output_shape[-1], input_.shape[-1]], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        conv_t = tf.nn.conv2d_transpose(input_, w, output_shape, strides=[1, 2, 2, 1], padding = "SAME")
        bn = bnorm(conv_t + b)
        act = lrelu(bn)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def conv_tran_bn_sigmoid(input_, output_shape, name="conv_t", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [4, 4, output_shape[-1], input_.shape[-1]], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        conv_t = tf.nn.conv2d_transpose(input_, w, output_shape, strides=[1, 2, 2, 1], padding = "SAME")
        bn = bnorm(conv_t + b)
        act = tf.nn.sigmoid(bn)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def conv_tran_bn_tanh(input_, output_shape, name="conv_t", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [4, 4, output_shape[-1], input_.shape[-1]], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        conv_t = tf.nn.conv2d_transpose(input_, w, output_shape, strides=[1, 2, 2, 1], padding = "SAME")
        bn = bnorm(conv_t + b)
        act = tf.nn.tanh(bn)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def conv_tran_tanh(input_, output_shape, name="conv_t", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [4, 4, output_shape[-1], input_.shape[-1]], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        conv_t = tf.nn.conv2d_transpose(input_, w, output_shape, strides=[1, 2, 2, 1], padding = "SAME")
        act = tf.nn.tanh(conv_t + b)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def dense_sigmoid(input_, neurons, name="dense", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [input_.shape[1], neurons], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        act = tf.nn.sigmoid(tf.add(tf.matmul(input_, w), b))
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def dense_bn_relu(input_, neurons, name="dense", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [input_.shape[1], neurons], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        bn = bnorm(tf.add(tf.matmul(input_, w), b))
        act = tf.nn.relu(bn)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def dense_bn_lrelu(input_, neurons, name="dense", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [input_.shape[1], neurons], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        bn = bnorm(tf.add(tf.matmul(input_, w), b))
        act = lrelu(bn)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act

def dense_bn_sigmoid(input_, neurons, name="dense", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("W", shape = [input_.shape[1], neurons], initializer = tf.random_normal_initializer())
        b = tf.get_variable("B", shape = [1], initializer = tf.zeros_initializer())
        bn = bnorm(tf.add(tf.matmul(input_, w), b))
        act = tf.nn.sigmoid(bn)
        if 'USE_TENSORBOARD' in globals() and USE_TENSORBOARD:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        return act
