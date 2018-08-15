import tensorflow as tf
import tensorflow.contrib.slim as slim


class VAE(object):
    latent_size = 2
    
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        batch_size = tf.shape(self.x)[0]
        
        z_mean, z_log_var = VAE.encoder_network(self.x)
        eps = tf.random_normal([batch_size, VAE.latent_size], mean=0., stddev=1e-2)
        z = z_mean + tf.exp(z_log_var) * eps
        self.z = z
        
        logits = VAE.decoder_network(z)
        self.out = tf.nn.sigmoid(logits)

        labels_flat = tf.reshape(self.x, [batch_size, -1])
        logits_flat = tf.reshape(logits, [batch_size, -1])
        
        image_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_flat, logits=logits_flat)
        image_loss = tf.reduce_sum(image_loss, axis=-1)
        # image_loss = tf.reduce_sum(tf.squared_difference(labels_flat, logits_flat), axis=-1)
        
        kl_loss = -0.5 * tf.reduce_sum(1.0 + 2.0*z_log_var - tf.square(z_mean) - tf.exp(2.0*z_log_var), 1)
        self.loss = tf.reduce_mean(image_loss + kl_loss)
        
    def encoder_network(x):
        net = slim.conv2d(x, 64, [2, 2], scope='conv1')
        net = slim.conv2d(net, 64, [2, 2], stride=2, scope='conv2')
        net = slim.conv2d(net, 64, [1, 1], scope='conv3')
        net = slim.conv2d(net, 64, [1, 1], scope='conv4')

        net = slim.flatten(net, scope='flat')
        net = slim.fully_connected(net, 128, scope='fc5')

        z_mean = slim.fully_connected(net, VAE.latent_size, scope='z_mean')
        z_log_var = slim.fully_connected(net, VAE.latent_size, scope='z_log_var')
        return z_mean, z_log_var

    def decoder_network(x):
        net = slim.fully_connected(x, 128, scope='fc1')
        net = slim.fully_connected(net, 128 * 14 * 14, scope='fc2')

        net = tf.reshape(net, [-1, 14, 14, 128])

        net = slim.conv2d_transpose(net, 64, [3, 3], scope='trans_conv3')
        net = slim.conv2d_transpose(net, 64, [3, 3], scope='trans_conv4')
        net = slim.conv2d_transpose(net, 64, [2, 2], stride=2, padding='VALID', scope='trans_conv5')
        net = slim.conv2d_transpose(net, 1, [2, 2], activation_fn=None, scope='trans_conv6')
        return net
