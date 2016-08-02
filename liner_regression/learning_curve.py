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

DEGREE = 1
LAMBDA = 0
LEARNING_RATE = 0.005
EPOCHES = 2000

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

placeholder_X = tf.placeholder("float32", [None, DEGREE])
placeholder_Y = tf.placeholder("float32", [None, 1])
W = tf.Variable(initial_value=tf.constant(1.0, shape=[DEGREE, 1]), name="weight")
b = tf.Variable(initial_value=[1.0], name="bias")

# construct a polynomial model

h = tf.add(tf.matmul(placeholder_X, W), b)

regulation_term = tf.reduce_mean(tf.square(W))*LAMBDA
h_error = tf.reduce_mean(tf.square(placeholder_Y - h))
cost = h_error + regulation_term # objective function
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    trainingset_error_list = []
    validationset_error_list = []

    for n in range(len(X)):
        print("Start Training N=%d" % n)
        for epoch in range(EPOCHES):
            sess.run(optimizer, feed_dict={placeholder_X: X2Vec(X, DEGREE)[:n], placeholder_Y: y[:n]})

        trainingset_error = sess.run(h_error, feed_dict={placeholder_X: X2Vec(X, DEGREE)[:n], placeholder_Y: y[:n]})
        validationset_error = sess.run(h_error, feed_dict={placeholder_X: X2Vec(Xval, DEGREE)[:n], placeholder_Y: yval[:n]})
        trainingset_error_list.append(trainingset_error)
        validationset_error_list.append(validationset_error)


    plt.plot(range(len(X)), trainingset_error_list, "bs", label="Training Error")
    plt.plot(range(len(X)), validationset_error_list, "ro", label="Validation Error")
    plt.legend()
    plt.show()

