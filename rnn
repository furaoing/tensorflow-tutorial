import tensorflow as tf

num_units = 1

x = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([1, 1]))
state = tf.zeros([1])
probabilities = []
loss = 0.0
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units)
cc = rnn_cell(x, state)
