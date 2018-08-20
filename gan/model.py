import tensorflow as tf
import tensorflow.contrib.slim as slim


class GAN(object):
    def __init__(self, smooth=0.9):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.z = tf.placeholder(tf.float32, [None, 128])
        self.keep_prob = tf.placeholder(tf.float32)

        g_logits = self.generator(self.z)

        d_logits_fake = self.discriminator(g_logits)
        d_logits_real = self.discriminator(self.x)

        self.g_loss = self.generator_loss(d_logits_fake)
        self.d_loss = self.discriminator_loss(d_logits_fake, d_logits_real, smooth)

        with tf.name_scope('summaries'):
            tf.summary.scalar('g_loss', self.g_loss)
            tf.summary.scalar('d_loss', self.d_loss)
        self.summary = tf.summary.merge_all()

    def generator(self, inputs, alpha=0.01):
        with tf.variable_scope('generator'):
            net = slim.fully_connected(inputs, 512, activation_fn=None, scope='fc1')
            net = tf.nn.leaky_relu(net, alpha=alpha)

            net = slim.fully_connected(net, 14 * 14, activation_fn=None, scope='fc2')
            net = tf.nn.leaky_relu(net, alpha=alpha)

            net = tf.reshape(net, [-1, 14, 14, 1])

            net = slim.conv2d(net, 128, [3, 3], scope='conv3')
            net = slim.conv2d_transpose(net, 128, [3, 3], stride=2, scope='conv4')
            net = slim.conv2d(net, 1, [3, 3], activation_fn=None, scope='conv5')
        return net

    def generator_loss(self, logits_fake):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake)
        return tf.reduce_mean(loss)

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            net = slim.conv2d(inputs, 256, [5, 5], activation_fn=tf.nn.leaky_relu, scope='conv1')
            net = slim.conv2d(net, 512, [5, 5], activation_fn=tf.nn.leaky_relu, scope='conv2')

            net = slim.flatten(net, scope='flat')
            net = slim.dropout(net, keep_prob=self.keep_prob, scope='dropout2')

            net = slim.fully_connected(net, 256, activation_fn=tf.nn.leaky_relu, scope='fc3')
            net = slim.dropout(net, keep_prob=self.keep_prob, scope='dropout3')

            net = slim.fully_connected(net, 2, activation_fn=None, scope='fc4')
        return net

    def discriminator_loss(self, logits_fake, logits_real, smooth):
        labels_fake = tf.zeros_like(logits_fake)
        labels_real = tf.ones_like(logits_real) * (1 - smooth)

        loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake)
        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_real)
        return tf.reduce_mean(loss_fake + loss_real)

