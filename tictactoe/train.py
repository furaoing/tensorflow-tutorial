import tensorflow as tf
from waffle import system
import pandas as pd

names = [str(x) for x in range(1, 10)]
names.append("label")
data_path = system.abs_path("tic-tac-toe.data")
dataset = pd.read_table(data_path, sep=r",", names=names)

train_x = dataset.iloc[:, 0:9]
train_y = dataset["label"]

mapping_lookup = {"x": 0, "o": 1, "b": 2, "positive": 1, "negative": 0}

train_x_list = train_x.values.tolist()
for i in range(len(train_x_list)):
    for j in range(len(train_x_list[i])):
        train_x_list[i][j] = mapping_lookup[train_x_list[i][j]]

train_y_list = train_y.values.tolist()
for i in range(len(train_y)):
    train_y_list[i] = mapping_lookup[train_y_list[i]]

onehot_depth = 3

train_x_onehot = tf.one_hot(train_x_list, onehot_depth)

train_x_onehot = tf.reshape(train_x_onehot, [len(train_x), 27])
train_y_list = [[x] for x in train_y_list]

n_hidden_unit_per_layer = 5
n_feature = 9*onehot_depth
learning_rate = 1
epoch = 5000

X = tf.placeholder(tf.float32, [None, n_feature])
W = tf.get_variable("W", [n_feature, n_hidden_unit_per_layer], initializer=tf.random_uniform_initializer())
b = tf.get_variable("b", [n_hidden_unit_per_layer])
y_ = tf.placeholder(tf.float32, [None, 1])

output = tf.nn.sigmoid(tf.matmul(X, W) + b)
loss = tf.reduce_mean(tf.square(y_ - output))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x_onehot_eval = sess.run(train_x_onehot)

    for i in range(epoch):
        _, result = sess.run([optimizer, loss], feed_dict={X: train_x_onehot_eval, y_: train_y_list})
        print("Loss: %f" % result)

    correct_prediction = tf.equal(tf.round(output), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={X: train_x_onehot_eval, y_: train_y_list})
    print("Accuracy: %f" % acc)

