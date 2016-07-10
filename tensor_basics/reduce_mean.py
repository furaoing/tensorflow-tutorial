# -*- coding: utf-8 -*-

import tensorflow as tf

a = tf.constant([[4, 4, 4], [3, 3, 3]])

op1 = tf.reduce_mean(a, reduction_indices=1)

with tf.Session() as sess:
    result = sess.run(op1)
    print(result)