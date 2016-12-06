import tensorflow as tf
import numpy as np

vocabulary_size = 600
embedding_size = 10

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
