import tensorflow as tf
import tensorflow.contrib.slim as slim

h, w, c = 28, 28, 3


class MNIST(object):
    n_classes = 10
    n_colors = 7

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, h, w, c])
        self.y_class = tf.placeholder(tf.int64, [None])
        self.y_color = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        self.conv = self.conv_layer()
        self.class_logits = self.fc_layer_class()
        self.color_logits = self.fc_layer_color()
        
        class_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_logits, labels=self.y_class)
        class_loss = tf.reduce_mean(class_cross_entropy)

        color_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.color_logits, labels=self.y_color)
        color_loss = tf.reduce_mean(color_cross_entropy)

        self.loss = class_loss + color_loss
        
        self.class_out = tf.argmax(tf.nn.softmax(self.class_logits), axis=-1)
        self.color_out = tf.argmax(tf.nn.softmax(self.color_logits), axis=-1)

    def conv_layer(self):
        with slim.arg_scope([slim.conv2d], kernel_size=[5, 5]), \
             slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]):
            net = slim.conv2d(self.x, 32, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')

            net = slim.conv2d(net, 64, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')
        return net

    def fc_layer_class(self):
        net = slim.flatten(self.conv, scope='class_flat')
        with slim.arg_scope([slim.fully_connected], num_outputs=1024), \
             slim.arg_scope([slim.dropout], keep_prob=self.keep_prob):
            net = slim.fully_connected(net, scope='class_fc1')
            net = slim.dropout(net, scope='class_dropout1')

        net = slim.fully_connected(net, self.n_classes, activation_fn=None, scope='class_out')
        return net

    def fc_layer_color(self):
        net = slim.flatten(self.conv, scope='color_flat')
        with slim.arg_scope([slim.fully_connected], num_outputs=1024), \
             slim.arg_scope([slim.dropout], keep_prob=self.keep_prob):
            net = slim.fully_connected(net, scope='color_fc1')
            net = slim.dropout(net, scope='color_dropout1')

        net = slim.fully_connected(net, self.n_colors, activation_fn=None, scope='color_out')
        return net

