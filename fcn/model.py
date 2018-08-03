import tensorflow as tf
import tensorflow.contrib.slim as slim
from upsampling import bilinear_upsample_weights


class FCN(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

        self.inputs = tf.placeholder(tf.float32, [None, None, None, 3])
        self.masks = tf.placeholder(tf.int64, [None, None, None])
        self.keep_prob = tf.placeholder(tf.float32)

        VGG_MEAN = tf.reshape(tf.constant([123.68, 116.78, 103.94]), [1, 1, 1, 3])
        logits = self.network(self.inputs - VGG_MEAN)
        self.softmax = tf.nn.softmax(logits)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.masks)
        self.loss = tf.reduce_mean(cross_entropy)
        
    def network(self, x):
        with tf.variable_scope('vgg_16'):
            net, vgg_end_points = self.vgg_conv(x)
        with tf.variable_scope('fcn_2s'):
            net = self.vgg_fc(net)
            net = self.fcn_net(net, vgg_end_points)
        return net

    def vgg_conv(self, x):
        end_points_collection = 'vgg_16'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=end_points_collection):
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
            
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points
    
    def vgg_fc(self, x):
        net = slim.conv2d(x, 4096, [1, 1], scope='fc6')
        net = slim.dropout(net, self.keep_prob, scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, self.keep_prob, scope='dropout7')
        net = slim.conv2d(net, self.n_classes, [1, 1],
                          activation_fn=None, normalizer_fn=None, scope='fc8')
        return net
        
    def fcn_net(self, x, end_points):
        def _shape(tensor):
            shape = tf.shape(tensor)
            return shape[0], shape[1], shape[2], shape[3]
        
        n_classes = self.n_classes
        upsample_filter_2 = tf.constant(bilinear_upsample_weights(factor=2, n_classes=n_classes))
        
        with slim.arg_scope([slim.conv2d],
                            activation_fn=None,
                            normalizer_fn=None,
                            weights_initializer=tf.zeros_initializer):
            # first upsampling x2
            n, h, w, c = _shape(x)
            output_shape = tf.stack([n, h*2, w*2, c])
            # 2x conv7
            net = tf.nn.conv2d_transpose(x, upsample_filter_2,
                                         output_shape=output_shape, strides=[1, 2, 2, 1])
            # pool4
            pool4_features = end_points['vgg_16/pool4']
            pool4_logits = slim.conv2d(pool4_features, n_classes, [1, 1], scope='pool4_fc')
            # pool4 + 2x conv7
            net = pool4_logits + net

            # second upsampling x2
            n, h, w, c = _shape(net)
            output_shape = tf.stack([n, h*2, w*2, c])
            # 2x (pool4 + 2x conv7)
            net = tf.nn.conv2d_transpose(net, upsample_filter_2,
                                         output_shape=output_shape, strides=[1, 2, 2, 1])
            # pool3
            pool3_features = end_points['vgg_16/pool3']
            pool3_logits = slim.conv2d(pool3_features, n_classes, [1, 1], scope='pool3_fc')
            # pool3 + 2x pool4 + 4x conv7
            net = pool3_logits + net

            # third upsampling x2
            n, h, w, c = _shape(net)
            output_shape = tf.stack([n, h*2, w*2, c])
            # 2x (pool3 + 2x pool4 + 4x conv7)
            net = tf.nn.conv2d_transpose(net, upsample_filter_2,
                                        output_shape=output_shape, strides=[1, 2, 2, 1])
            # pool2
            pool2_features = end_points['vgg_16/pool2']
            pool2_logits = slim.conv2d(pool2_features, n_classes, [1, 1], scope='pool2_fc')
            # pool2 + 2x pool3 + 4x pool4 + 8x pool7
            net = pool2_logits + net
            
            # fourth upsampling x2
            n, h, w, c = _shape(net)
            output_shape = tf.stack([n, h*2, w*2, c])
            # 2x (2x pool3 + 4x pool4 + 8x pool7)
            net = tf.nn.conv2d_transpose(net, upsample_filter_2,
                                         output_shape=output_shape, strides=[1, 2, 2, 1])
            # pool1
            pool1_features = end_points['vgg_16/pool1']
            pool1_logits = slim.conv2d(pool1_features, n_classes, [1, 1], scope='pool1_fc')
            # pool1 + 2x pool2 + 4x pool3 + 8x pool4 + 16x pool7
            net = pool1_logits + net

            # final upsampling x2
            n, h, w, c = _shape(net)
            output_shape = tf.stack([n, h*2, w*2, c])
            net = tf.nn.conv2d_transpose(net, upsample_filter_2,
                                         output_shape=output_shape, strides=[1, 2, 2, 1])
        return net

if __name__ == '__main__':
    FCN(10)