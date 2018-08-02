import tensorflow as tf
import tensorflow.contrib.slim as slim

h, w, c = 28, 28, 1


class Siamese(object):
    n_classes = 2

    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, h, w, c])
        self.x2 = tf.placeholder(tf.float32, [None, h, w, c])
        self.y_ = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('siamese') as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        logits = tf.abs(self.o1 - self.o2)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits)
        self.loss = tf.reduce_mean(cross_entropy)
        self.out = tf.nn.sigmoid(logits)

    def network(self, x):
        with slim.arg_scope([slim.conv2d], kernel_size=[5, 5]), \
             slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]):
            net = slim.conv2d(x, 32, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')

            net = slim.conv2d(net, 64, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')

        net = slim.flatten(net, scope='flat')

        with slim.arg_scope([slim.fully_connected], num_outputs=1024), \
             slim.arg_scope([slim.dropout], keep_prob=self.keep_prob):
            net = slim.fully_connected(net, scope='fc1')
            net = slim.dropout(net, scope='dropout1')

        net = slim.fully_connected(net, self.n_classes, activation_fn=None, scope='fc2')
        return net

