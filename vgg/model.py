import tensorflow as tf
import tensorflow.contrib.slim as slim


class VGG(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])

        rgb_mean_t = tf.reshape(tf.constant([123.68, 116.78, 103.94]), [1, 1, 1, 3])
        with tf.variable_scope('vgg_16'):
            logits = tf.squeeze(self.network(self.x - rgb_mean_t), axis=[1, 2])
        self.softmax = tf.nn.softmax(logits)

    def network(self, x):
        net = slim.repeat(x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # use conv2d instead of fully_connected layers
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, is_training=False, scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, is_training=False, scope='dropout7')
        net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, scope='fc8')
        return net

