import tensorflow as tf
import tensorflow.contrib.slim as slim


class GAN(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.z = tf.placeholder(tf.float32, [None, 100])
        self.keep_prob = tf.placeholder(tf.float32)

        g = self.generator(self.z, self.keep_prob)
        d_fake = self.discriminator(g, self.keep_prob)
        d_real = self.discriminator(self.x, self.keep_prob)

        self.g_loss = self.generator_loss(d_fake)
        self.d_loss = self.discriminator_loss(d_fake, d_real)
        self.inference = g
        
        tf.summary.histogram('g', g)
        tf.summary.histogram('d_fake', d_fake)
        tf.summary.histogram('d_real', d_real)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        self.summary = tf.summary.merge_all()

    def generator(self, inputs, keep_prob):
        with tf.variable_scope('generator'):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.leaky_relu):
                net = slim.fully_connected(inputs, 7*7*256, scope='fc1')
                net = slim.batch_norm(net, scope='bn1')

            net = tf.reshape(net, [-1, 7, 7, 256], name='reshape')
            net = slim.dropout(net, keep_prob, scope='dropout1')

            with slim.arg_scope([slim.conv2d_transpose],
                                kernel_size=[5, 5],
                                stride=2,
                                activation_fn=tf.nn.leaky_relu):
                net = slim.conv2d_transpose(net, 128, scope='conv2')
                net = slim.batch_norm(net, scope='bn2')

                net = slim.conv2d_transpose(net, 64, scope='conv3')
                net = slim.batch_norm(net, scope='bn3')

            net = slim.conv2d_transpose(net, 1, [5, 5], stride=1,
                                        activation_fn=tf.nn.sigmoid, scope='conv4')
        return net

    def discriminator(self, inputs, keep_prob):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d],
                                kernel_size=[5, 5],
                                activation_fn=tf.nn.leaky_relu):
                net = slim.conv2d(inputs, 64, stride=2, scope='conv1')
                net = slim.dropout(net, keep_prob, scope='dropout1')
                
                net = slim.conv2d(net, 64, scope='conv2')
                net = slim.dropout(net, keep_prob, scope='dropout2')
                
                net = slim.conv2d(net, 64, scope='conv3')
                net = slim.dropout(net, keep_prob, scope='dropout3')
            
            net = slim.flatten(net, scope='flat')
            net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='fc5')
            return net
        
    def binary_cross_entropy(self, labels, logits, eps=1e-12):
        z, x = labels, logits
        return z * -tf.log(x + eps) + (1. - z) * -tf.log(1. - x + eps)

    def generator_loss(self, logits_fake):
        loss = self.binary_cross_entropy(labels=tf.ones_like(logits_fake), logits=logits_fake)
        return tf.reduce_mean(loss)

    def discriminator_loss(self, logits_fake, logits_real):
        labels_fake = tf.zeros_like(logits_fake)
        labels_real = tf.ones_like(logits_real)

        loss_fake = self.binary_cross_entropy(labels=labels_fake, logits=logits_fake)
        loss_real = self.binary_cross_entropy(labels=labels_real, logits=logits_real)
        return tf.reduce_mean(0.5 * (loss_fake + loss_real))
