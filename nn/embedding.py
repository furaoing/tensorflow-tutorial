# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

dataset = tf.random_normal([500, 10])

learning_rate = 0.000000000001
training_epochs = 5000
display_step = 1

n_hidden_1 = 50
n_hidden_2 = 50
n_input = 10
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(tf.constant(1.0, shape=[n_input, n_hidden_1])),
    'h2': tf.Variable(tf.constant(1.0, shape=[n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.constant(1.0, shape=[n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.constant(1.0, shape=[n_hidden_1])),
    'b2': tf.Variable(tf.constant(1.0, shape=[n_hidden_2])),
    'out': tf.Variable(tf.constant(1.0, shape=[n_classes])),
}


def multiplayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


proc = multiplayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(proc, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.initialize_all_variables()

a = time.time()
with tf.Session() as sess:
    with tf.device("/cpu"):
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0

            _, c = sess.run([optimizer, cost], feed_dict={x: dataset.eval(),
                                                          y: dataset.eval()})

            avg_cost += c

            if epoch % display_step == 0:
                print("Epoch: %d, cost=%f" % (epoch+1, avg_cost))
        print("Optimization Finished")

        correct_prediction = tf.equal(tf.argmax(proc, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy: %f" % (accuracy.eval({x: dataset.eval(), y: dataset.eval()})))
b = time.time()
print("Time Elapsed: %f" % (b-a))
