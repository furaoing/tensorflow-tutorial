import random
import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn

random.seed(42)

X_train = ["s", "a", "s", "d"]
cat_processor = learn.preprocessing.CategoricalProcessor()
X_train = np.array(list(cat_processor.fit_transform(X_train)))
ids = [x[0] for x in X_train]

input_ids = tf.placeholder(dtype=tf.int32, shape=[None])

embedding = tf.Variable(np.identity(4, dtype=np.int32))
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
# can use tf.one_hot()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

c = sess.run(input_embedding, feed_dict={input_ids: ids})
