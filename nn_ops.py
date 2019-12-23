from .ImageClassification import *
import numpy as np

def model(x):
    x = tf.cast(x, dtype=tf.float32)
    c1 = conv2d(x, weights[0], stride_size=1)
    c1 = conv2d(c1, weights[1], stride_size=1)
    p1 = maxpool(c1, pool_size=2, stride_size=2)

    c2 = conv2d(p1, weights[2], stride_size=1)
    c2 = conv2d(c2, weights[3], stride_size=1)
    p2 = maxpool(c2, pool_size=2, stride_size=2)

    c3 = conv2d(p2, weights[4], stride_size=1)
    c3 = conv2d(c3, weights[5], stride_size=1)
    p3 = maxpool(c3, pool_size=2, stride_size=2)

    c4 = conv2d(p3, weights[6], stride_size=1)
    c4 = conv2d(c4, weights[7], stride_size=1)
    p4 = maxpool(c4, pool_size=2, stride_size=2)

    c5 = conv2d(p4, weights[8], stride_size=1)
    c5 = conv2d(c5, weights[9], stride_size=1)
    p5 = maxpool(c5, pool_size=2, stride_size=2)

    c6 = conv2d(p5, weights[10], stride_size=1)
    c6 = conv2d(c6, weights[11], stride_size=1)
    p6 = maxpool(c6, pool_size=2, stride_size=2)

    flatten = tf.reshape(p6, shape=(tf.shape(p6)[0], -1))

    d1 = dense(flatten, weights[12])
    d2 = dense(d1, weights[13])
    d3 = dense(d2, weights[14])
    d4 = dense(d3, weights[15])
    d5 = dense(d4, weights[16])
    logits = tf.matmul(d5, weights[17])

    return tf.nn.softmax(logits)


def loss( pred , target ):
    return tf.losses.categorical_crossentropy( target , pred )


optimizer = tf.optimizers.Adam(0.5)


def train_step(model, inputs, outputs):
    with tf.GradientTape() as tape:
        current_loss = loss(model(inputs), outputs)
    grads = tape.gradient(current_loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    print(tf.reduce_mean(current_loss))


num_epochs = 5

for e in range(num_epochs):
    for features in dataset:
        image, label = features['image'], features['label']
        train_step(model, image, tf.one_hot(label, depth=3))

