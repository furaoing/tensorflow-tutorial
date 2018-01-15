#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:06:33 2018

@author: rao
"""

import tensorflow as tf
import random

ran = [0, 1]
a = [random.choice(ran) for x in range(10000)]
b = [random.choice(ran) for x in range(10000)]

c = [x*y for x, y in zip(a, b)]
train_x = list(zip(a, b))
train_y = [[x] for x in c]

size = len(c)

n_hidden_unit_per_layer = 1
n_feature = 2
learning_rate = 0.5
epoch = 3000

X = tf.placeholder(tf.float32, [None, n_feature])
W = tf.get_variable("W", [n_feature, n_hidden_unit_per_layer], initializer=tf.random_uniform_initializer())
b = tf.get_variable("b", [n_hidden_unit_per_layer])
y_ = tf.placeholder(tf.float32, [None, 1])

output = tf.nn.sigmoid(tf.matmul(X, W) + b)
loss = tf.reduce_mean(tf.square(y_ - output))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch):
        _, result = sess.run([optimizer, loss], feed_dict={X: train_x, y_: train_y})
        print(result)


    


