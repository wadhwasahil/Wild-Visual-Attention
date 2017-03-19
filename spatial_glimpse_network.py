import tensorflow as tf
import numpy as np


class SGN(object):
    def __init__(self, w=32, filter_size=5, num_filters=96):
        self.input_x = tf.placeholder(tf.float32, [None, w, w, 3], name="X_train")
        with tf.name_scope("conv1"):
            pad_arr = [[0, 0], [2, 2], [2, 2], [0, 0]]
            self.X = tf.pad(self.input_x, pad_arr, mode="CONSTANT")
            filter_shape = [filter_size, filter_size, 3, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                self.X,
                W,
                strides=[1, 2, 2, 1],
                padding="VALID",
                name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            h = tf.pad(h, pad_arr, mode="CONSTANT")
            self.pool = tf.nn.max_pool(
                h,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name="pool")