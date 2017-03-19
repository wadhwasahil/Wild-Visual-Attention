import read_data
import tensorflow as tf
import numpy as np
import spatial_glimpse_network

data = read_data.read()

img = next(data)
img = np.expand_dims(np.resize(img, (32, 32, 3)), 0)
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = spatial_glimpse_network.SGN()
        sess.run(tf.global_variables_initializer())
        pool = sess.run([cnn.pool], feed_dict={cnn.input_x:img})


