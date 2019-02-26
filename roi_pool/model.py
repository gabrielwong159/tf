import tensorflow as tf
import tensorflow.contrib.slim as slim


class Model(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 128, 128, 1])
        self.y = tf.placeholder(tf.float32, [None, 2])

        logits = self.network(self.x)
        mse = tf.reduce_mean(tf.square(logits - self.y))
        self.loss = mse
        self.inference = logits

        with tf.name_scope('inputs'):
            tf.summary.histogram('image', self.x)
            tf.summary.histogram('label', self.y)
        with tf.name_scope('metrics'):
            tf.summary.scalar('mse', mse)
        self.summary = tf.summary.merge_all()

    def network(self, x):
        net = slim.conv2d(x, 32, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  # [64, 64]

        net = slim.conv2d(net, 64, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')  # [32, 32]

        net = slim.conv2d(net, 64, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')  # [16, 16]

        net = slim.conv2d(net, 128, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')  # [8, 8]

        net = slim.flatten(net)
        net = slim.fully_connected(net, 256, scope='fc5')
        net = slim.fully_connected(net, 2, activation_fn=None, scope='fc6')
        return net

