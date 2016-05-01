# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:39:57 2016

@author: rao
"""

import tensorflow as tf

sess = tf.Session()


list_a = tf.constant([1, 3, 4, 2])

argmax = tf.argmax(list_a, dimension=0)

result = sess.run(argmax)

print(result)
