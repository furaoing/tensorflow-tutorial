# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
rng = np.random
x_data = tf.constant(1, 'float32', [13, 3])
y_data = tf.constant(1, 'float32', [3, 1])


op = tf.matmul(x_data, y_data)

with tf.Session() as sess:
    result = sess.run(op)
    print(result)
