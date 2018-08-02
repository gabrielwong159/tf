import tensorflow as tf
import tensorflow.contrib.slim as slim


class MNIST(object):
    h, w, c = 28, 28, 1
    n_classes = 10

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.h, self.w, self.c])
        self.y = tf.placeholder(tf.int64, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32)

        logits = self.network()
        self.softmax = tf.nn.softmax(logits)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        self.loss = tf.reduce_mean(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(self.softmax, 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def network(self):
        with slim.arg_scope([slim.conv2d], kernel_size=[5, 5]), \
             slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]):
            net = slim.conv2d(self.x, 32, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')

            net = slim.conv2d(net, 64, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')

        net = slim.flatten(net, scope='flat')

        with slim.arg_scope([slim.fully_connected], num_outputs=1024), \
             slim.arg_scope([slim.dropout], keep_prob=self.keep_prob):
            net = slim.fully_connected(net, scope='fc1')
            net = slim.dropout(net, scope='dropout1')

        net = slim.fully_connected(net, self.n_classes, activation_fn=None, scope='out')
        return net

