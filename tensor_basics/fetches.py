# -*- coding: utf-8 -*-

import tensorflow as tf

input1 = tf.constant(1.0)
input2 = tf.constant(2.0)
input3 = tf.constant(3.0)

intermediate = tf.add(input1, input2)
mul = tf.mul(input3, intermediate)

with tf.Session() as sess:
    result = sess.run([intermediate, mul])

print(result)
