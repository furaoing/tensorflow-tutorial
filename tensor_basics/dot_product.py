# -*- coding: utf-8 -*-

import tensorflow as tf

a = tf.constant([[1, 2, 3]])
b = tf.constant([[1, 2, 3]])


proc = tf.matmul(a, tf.transpose(b))
#proc = tf.matmul(a, b)


with tf.Session() as sess:
    result = sess.run(proc)
    print(result)
