# -*- coding: utf-8 -*-

import tensorflow as tf

a = tf.constant([1, 2, 3], dtype=tf.float32)

b = a + [0, 0, 1]

with tf.Session() as sess:
    result = sess.run(b)

print(result)
