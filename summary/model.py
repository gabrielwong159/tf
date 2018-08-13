import tensorflow as tf
import tensorflow.contrib.slim as slim


class MNIST(object):
    h, w, c = 28, 28, 1
    n_classes = 10

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.h, self.w, self.c])
        self.y = tf.placeholder(tf.int64, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32)

        self.logits = self.network()
        self.softmax = tf.nn.softmax(self.logits)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(self.softmax, 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def network(self):
        net = slim.conv2d(self.x, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.flatten(net, scope='flat')

        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.dropout(net, self.keep_prob, scope='dropout1')

        net = slim.fully_connected(net, self.n_classes, activation_fn=None, scope='fc2')
        return net
