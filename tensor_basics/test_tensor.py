# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

x_data = tf.constant(1, 'float32')

with tf.Session() as sess:
    result = sess.run(x_data)
    print(x_data)
