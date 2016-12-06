# -*- coding: utf-8 -*-

from scipy.io import loadmat
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import copy


def plot_data_set(X, y, Xtest, ytest, Xval, yval):
    plt.plot(X, y, 'ro', label="Training Data")
    plt.legend()
    plt.show()

    plt.plot(Xtest, ytest, 'ro', label="Test Data")
    plt.legend()
    plt.show()

    plt.plot(Xval, yval, 'ro', label="Validation Data")
    plt.legend()
    plt.show()


mat_file_name = "ex5data1.mat"
mat_file_data = loadmat(mat_file_name)

X = mat_file_data.get("X")
y = mat_file_data.get("y")

Xtest = mat_file_data.get("Xtest")
ytest = mat_file_data.get("ytest")

Xval = mat_file_data.get("Xval")
yval = mat_file_data.get("yval")

#plot_data_set(X, y, Xtest, ytest, Xval, yval)

DEGREE = 8
LAMBDA = 0.001
LEARNING_RATE = 0.005
EPOCHES = 3000


def X2Vec(X, DEGREE):
    abs_X = np.absolute(X)
    max_index = np.argmax(abs_X)
    Xtrain_iter = map(lambda x: [math.pow(x, degree) for degree in range(1, DEGREE + 1)], X)
    Xtrain = list(Xtrain_iter)

    max_list = copy.deepcopy(Xtrain[max_index])
    for i in range(len(Xtrain)):
        for j in range(len(Xtrain[0])):
            Xtrain[i][j] = Xtrain[i][j]/abs(max_list[j])

    Xtrain = np.array(Xtrain)
    return Xtrain

a = X2Vec(X, DEGREE)
c = 1
placeholder_X = tf.placeholder("float32", [None, DEGREE])
placeholder_Y = tf.placeholder("float32", [None, 1])
W = tf.Variable(initial_value=tf.constant(1.0, shape=[DEGREE, 1]), name="weight")
b = tf.Variable(initial_value=[1.0], name="bias")

# construct a polynomial model

h = tf.add(tf.matmul(placeholder_X, W), b)

regulation_term = tf.reduce_mean(tf.square(W))*LAMBDA
cost = tf.reduce_mean(tf.square(placeholder_Y-h)) + regulation_term
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    cost_list = []

    for epoch in range(EPOCHES):
        sess.run(optimizer, feed_dict={placeholder_X: X2Vec(X, DEGREE), placeholder_Y: y})

        c = sess.run(cost, feed_dict={placeholder_X: X2Vec(X, DEGREE), placeholder_Y: y})
        predict = sess.run(h, feed_dict={placeholder_X: X2Vec(X, DEGREE), placeholder_Y: y})
        cost_list.append(c)
        print("Cost: %f" % c)

    plt.plot(cost_list, 'ro', label="Cost Curve")
    plt.legend()
    plt.show()

    plt.plot(X, predict, "bs", label="Fitted Line")
    plt.plot(X, y, 'ro', label="Original Data Points")
    plt.legend()
    plt.show()

