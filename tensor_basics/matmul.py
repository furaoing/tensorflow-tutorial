# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

x_data = tf.constant(1, 'int8', [2, 1])
y_data = tf.constant(1, 'int8', [2, 1])

op = tf.matmul(x_data, tf.transpose(y_data))

with tf.Session as sess:
    result = sess.run(op)

    print(result)
