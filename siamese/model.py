import tensorflow as tf
import tensorflow.contrib.slim as slim

h, w, c = 28, 28, 1


class Siamese(object):
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, h, w, c])
        self.x2 = tf.placeholder(tf.float32, [None, h, w, c])
        self.y_ = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('siamese') as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        self.dist = tf.sqrt(tf.reduce_sum(tf.square(self.o1 - self.o2), axis=-1))
        self.loss = tf.reduce_mean(self.contrastive_loss())

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

            net = slim.fully_connected(net, scope='fc2')
        return net

    def contrastive_loss(self):
        margin = 1.0
        loss_pos = self.y_ * tf.square(self.dist)
        loss_neg = (1.0 - self.y_) * tf.square(tf.maximum(0.0, margin - self.dist))
        return loss_pos + loss_neg
