"""
Pekka Aalto 2017

This snippet tries to explain by example what deepmind means
in https://arxiv.org/abs/1708.04782
about embedding on channel axis being equivalent to
one-hot-encoding followed by 1x1 conv.

They write:

"We embed all feature layers containing categorical values
into a continuous space which is equivalent to using
a one-hot encoding in the channel dimension
followed by a 1 ⇥ 1 convolution."

However, there are still two possible ways to do this. Either
1) concatenating
or
2) summing
the embeddings of each feature on channel dimension.

As mentioned in the deepmind-paper, both can be done as
- embedding lookup
or
- one-hot -> 1x1

However, we still don't know if they eventually concatenated or summed the embeddings.
"""

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

#### CONSTANTS
embedding_dim = 8
H = W = 32
n_cat_by_feature = [3, 10, 2]  # so 3 categorical features e.g player_relative, unit_type, is_creep

# will use this to set the weights for every category in every methodology
initial_emb_weights = [np.random.rand(n, embedding_dim) for n in n_cat_by_feature]

# the actual features
features = [
    tf.placeholder(shape=[H, W], dtype="int32", name="feat%d" % i)
    for i, _ in enumerate(n_cat_by_feature)
]

# 1.1) embed on channel -> concat on channel
embedded1 = []
for f, n, w in zip(features, n_cat_by_feature, initial_emb_weights):
    e = layers.embed_sequence(
        f,
        vocab_size=n,
        embed_dim=embedding_dim,
        initializer=tf.constant_initializer(w)
    )
    embedded1.append(e)

out11 = tf.concat(embedded1, axis=2)

# 1.2) onehot on channel -> 1x1 conv separately -> concat on channel
embedded2 = []
for f, n, w in zip(features, n_cat_by_feature, initial_emb_weights):
    one_hot = layers.one_hot_encoding(f, num_classes=n)

    conv_out = layers.conv2d(
        inputs=one_hot,
        num_outputs=embedding_dim,
        weights_initializer=tf.constant_initializer(w),
        kernel_size=1,
        stride=1
    )
    embedded2.append(conv_out)

out12 = tf.concat(embedded2, axis=2)

# 2.1) sum embeddings on channel instead of concatenating
out21 = tf.add_n(embedded1)

# 2.2) onehot on channel -> concat on channel -> 1x1 conv
one_hotted_features = tf.concat([
    layers.one_hot_encoding(f, num_classes=n)
    for f, n in zip(features, n_cat_by_feature)
], axis=2)

out22 = layers.conv2d(
    inputs=one_hotted_features,
    num_outputs=embedding_dim,
    weights_initializer=tf.constant_initializer(np.concatenate(initial_emb_weights)),
    kernel_size=1,
    stride=1
)

print(out11)
print(out12)
print(out21)
print(out22)

# let's try it:
feed_dict = {
    ph: np.random.randint(n, size=(H,W)) for ph,n in zip(features, n_cat_by_feature)
}

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

a11, a12, a21, a22 = sess.run([out11, out12, out21, out22], feed_dict=feed_dict)

#the result is indeed the same
assert np.all(np.abs(a11 - a12) < 1e-6)
assert np.all(np.abs(a21 - a22) < 1e-6)

#the result is not trivial
assert np.abs(a11).sum() > 1.0
assert np.abs(a21).sum() > 1.0

print("Phew it indeed worked. Good bye.")
