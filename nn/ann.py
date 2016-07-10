# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
mnist = input_data.read_data_sets("~/research_data/MNIST/", one_hot=True)

learning_rate = 0.05
training_epochs = 15
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

batch_x, batch_y = mnist.train.next_batch(100)

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes])),
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
    with tf.device("/gpu:0"):
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/batch_size)

            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})

                avg_cost += c/total_batch

            if epoch % display_step == 0:
                print("Epoch: %d, cost=%f" % (epoch+1, avg_cost))
        print("Optimization Finished")

        correct_prediction = tf.equal(tf.argmax(proc, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy: %f" % (accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))
b = time.time()
print("Time Elapsed: %f" % (b-a))
