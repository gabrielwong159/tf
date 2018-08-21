# https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/blob/master/tensorflow_MNIST_DCGAN.py
# https://github.com/ytakzk/Mnist-DCGAN-for-Tensorflow/blob/master/dcgan.ipynb
import tensorflow as tf
import tensorflow.contrib.slim as slim


class GAN(object):
    def __init__(self, smooth=0.9):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.z = tf.placeholder(tf.float32, [None, 100])
        self.keep_prob = tf.placeholder(tf.float32)

        g_logits = self.generator(self.z)
        self.out = tf.nn.sigmoid(g_logits)

        d_logits_fake = self.discriminator(g_logits)
        d_logits_real = self.discriminator(self.x)

        self.g_loss = self.generator_loss(d_logits_fake)
        self.d_loss = self.discriminator_loss(d_logits_fake, d_logits_real, smooth)

        with tf.name_scope('summaries'):
            tf.summary.scalar('g_loss', self.g_loss)
            tf.summary.scalar('d_loss', self.d_loss)
        self.summary = tf.summary.merge_all()

    def generator(self, inputs):
        with tf.variable_scope('generator'):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.leaky_relu):
                net = slim.fully_connected(inputs, 1024, scope='fc1')
                net = slim.batch_norm(net, scope='bn1')
                
                net = slim.fully_connected(net, 7*7*128, scope='fc2')
                net = slim.batch_norm(net, scope='bn2')
                
            net = tf.reshape(net, [-1, 7, 7, 128], name='reshape')

            with slim.arg_scope([slim.conv2d_transpose],
                                kernel_size=[5, 5],
                                stride=2,
                                activation_fn=tf.nn.leaky_relu):
                net = slim.conv2d_transpose(net, 64, scope='conv3')
                net = slim.batch_norm(net, scope='bn3')
                
                net = slim.conv2d_transpose(net, 1, activation_fn=None, scope='conv4')
        return net

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d],
                                kernel_size=[5, 5],
                                stride=2,
                                activation_fn=tf.nn.leaky_relu):
                net = slim.conv2d(inputs, 128, scope='conv1')
                
                net = slim.conv2d(net, 256, scope='conv2')
                net = slim.batch_norm(net, scope='bn2')
                
                net = slim.conv2d(net, 512, scope='conv3')
                net = slim.batch_norm(net, scope='bn3')
                
                net = slim.conv2d(net, 1024, scope='conv4')
                net = slim.batch_norm(net, scope='bn4')
                
                net = slim.conv2d(net, 1, stride=1, activation_fn=None, scope='conv5')
            return net
    
    def generator_loss(self, logits_fake):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake)
        return tf.reduce_mean(loss)

    def discriminator_loss(self, logits_fake, logits_real, smooth):
        labels_fake = tf.zeros_like(logits_fake)
        labels_real = tf.ones_like(logits_real) * (1 - smooth)

        loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake)
        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_real)
        return tf.reduce_mean(loss_fake + loss_real)
