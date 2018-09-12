# understanding fpn - https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
# tf implementation - https://github.com/yangxue0827/FPN_Tensorflow/blob/master/libs/rpn/build_rpn.py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils


class RPN(object):
    POS_ANCHOR_THRESH = 0.7
    NEG_ANCHOR_THRESH = 0.1

    h, w, c = 224, 224, 3

    anchor_scales = [64, 32, 16, 8]
    anchor_ratios = [0.5, 1.0, 2.0]
    backbone_shapes = np.array([
        [56, 56],
        [28, 28],
        [14, 14],
        [7, 7],
    ])
    feature_strides = [4, 8, 16, 32]


    def __init__(self):
        self.images = tf.placeholder(tf.float32, [None, self.h, self.w, self.c])
        self.gt_boxes = tf.placeholder(tf.float32, [None, None, 4])
        
        fpn = FPN(self.images)
        cls_logits, bbox_logits = self.build_rpn(fpn.layers)  # [batch, n_anchors, 2], [batch, n_anchors, 4]
        
        anchors = utils.generate_all_anchors(self.anchor_scales, self.anchor_ratios,
                                             self.backbone_shapes, self.feature_strides,
                                             anchor_stride=1, image_shape=[self.h, self.w])
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        
        cls_losses, bbox_losses = tf.map_fn(self.compute_loss,
                                            [self.gt_boxes, cls_logits, bbox_logits],
                                            dtype=(tf.float32, tf.float32))
        cls_loss = tf.reduce_mean(cls_losses)
        bbox_loss = tf.reduce_mean(bbox_losses)
        self.bbox_loss = bbox_loss
        self.loss = cls_loss + bbox_loss

        with tf.name_scope('summaries'):
            tf.summary.histogram('input', self.images)
            for layer_name in ['P5', 'P4', 'P3', 'P2']:
                tf.summary.histogram(layer_name, fpn.layers[layer_name])
            tf.summary.histogram('cls_logits', cls_logits)
            tf.summary.histogram('bbox_logits', bbox_logits)
            tf.summary.histogram('cls_probs', tf.nn.softmax(cls_logits))
            tf.summary.scalar('cls_loss', cls_loss)
            tf.summary.scalar('bbox_loss', bbox_loss)
            tf.summary.scalar('total_loss', self.loss)
        self.summaries = tf.summary.merge_all()
        
        apply_all_deltas = lambda logits: utils.apply_anchor_deltas(self.anchors, logits)
        self.bboxes = tf.map_fn(apply_all_deltas, bbox_logits, dtype=tf.float32)
        self.cls_probs = tf.nn.softmax(cls_logits)
        
        
        # remove gt_boxes used for padding
        is_valid = tf.reduce_all(self.gt_boxes[0] >= 0, axis=-1)
        indices = tf.where(is_valid)[:, 0]
        gt_boxes = tf.gather(self.gt_boxes[0], indices)
        anchors, labels, mappings = self.compute_anchor_labels(self.anchors, gt_boxes, cls_logits[0])
        bboxes = utils.apply_anchor_deltas(self.anchors, bbox_logits[0])
        self.test = (anchors, labels, mappings, bboxes)
        
    def rpn_fc(self, fpn, layer_name):
        net = slim.conv2d(fpn[layer_name], 512, [3, 3], scope=f'conv_rpn_{layer_name}')

        n_anchors = len(self.anchor_scales) * len(self.anchor_ratios)
        # {bg, fg}
        rpn_cls = slim.conv2d(net, n_anchors * 2, [1, 1], activation_fn=None, scope=f'fc_cls_{layer_name}')
        cls_logits = tf.reshape(rpn_cls, [tf.shape(rpn_cls)[0], -1, 2])  # [N, h, w, anchors*2] -> [N, anchors, 2]
        # [dy, dx, log(dh), log(dw)]
        rpn_box = slim.conv2d(net, n_anchors * 4, [1, 1], activation_fn=None, scope=f'fc_box_{layer_name}')
        bbox = tf.reshape(rpn_box, [tf.shape(rpn_box)[0], -1, 4])  # [N, h, w, anchors*4] -> [N, anchors, 4]
        return cls_logits, bbox
    
    def build_rpn(self, fpn):
        layers_outputs = [self.rpn_fc(fpn, layer_name) for layer_name in ['P5', 'P4', 'P3', 'P2']]
        cls_logits, bbox_logits = zip(*layers_outputs)
        cls_logits = tf.concat(cls_logits, axis=1)
        bbox_logits = tf.concat(bbox_logits, axis=1)
        return cls_logits, bbox_logits
    
    def compute_loss(self, args):
        gt_boxes, cls_logits, bbox_logits = args  # [n_gt_boxes, 4], [n_anchors, 2], [n_anchors, 4]
        # remove gt_boxes used for padding
        is_valid = tf.reduce_all(gt_boxes >= 0, axis=-1)
        indices = tf.where(is_valid)[:, 0]
        gt_boxes = tf.gather(gt_boxes, indices)
        
        # [6000, 4], [6000], [6000]
        top_k = self.get_top_k(self.anchors, cls_logits)
        anchors = tf.gather(self.anchors, top_k)
        cls_logits = tf.gather(cls_logits, top_k)
        bbox_logits = tf.gather(bbox_logits, top_k)

        anchors, labels, mappings = self.compute_anchor_labels(anchors, gt_boxes, cls_logits)
        cls_loss = self.compute_class_loss(labels, cls_logits)
        bbox_loss = self.compute_bbox_loss(anchors, labels, mappings, gt_boxes, bbox_logits)
        return cls_loss, bbox_loss
    
    def get_top_k(self, anchors, cls_logits):
        anchor_scores = tf.nn.softmax(cls_logits)[:, 1]  # [n_anchors]
        limit = tf.minimum(6000, tf.shape(anchors)[0])
        top_k_idx = tf.nn.top_k(anchor_scores, k=limit).indices
        return top_k_idx
        
    def compute_anchor_labels(self, anchors, gt_boxes, cls_logits):
        # compute ious for anchors against given ground truth boxes
        ious = utils.compute_ious(anchors, gt_boxes)  # [n_anchors, n_gt_boxes]
        labels = tf.zeros(shape=[tf.shape(anchors)[0]], dtype=tf.float32)
        mappings = tf.argmax(ious, axis=1)
        
        max_iou_per_anchor = tf.reduce_max(ious, axis=1)
        max_iou_per_gt_box = tf.reduce_max(ious, axis=0)

        pos1 = max_iou_per_anchor >= self.POS_ANCHOR_THRESH
        pos2 = tf.reduce_sum(tf.cast(tf.equal(ious, max_iou_per_gt_box), tf.float32), axis=1)  # anchors with largest iou per gt box
        pos = pos1 | tf.cast(pos2, tf.bool)
        labels += tf.cast(pos, tf.float32)

        neg = max_iou_per_anchor < self.NEG_ANCHOR_THRESH
        labels -= tf.cast(neg, tf.float32)
        return anchors, labels, mappings

    def compute_class_loss(self, anchor_labels, cls_logits):
        anchor_classes = tf.cast(tf.equal(anchor_labels, 1), tf.int64)  # map {-1, 0}/+1 to 0/1

        pos_indices = tf.where(tf.equal(anchor_labels, 1))[:, 0]
        # sub-sample number of negative anchors
        n_neg = tf.reduce_sum(anchor_classes) * 2
        neg_indices = tf.where(tf.equal(anchor_labels, -1))[:, 0]
        neg_indices = tf.random_shuffle(neg_indices)[:n_neg]
        indices = tf.concat([pos_indices, neg_indices], axis=0)
        
        anchor_classes = tf.gather(anchor_classes, indices)
        cls_logits = tf.gather(cls_logits, indices)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=anchor_classes, logits=cls_logits)
        loss = tf.maximum(loss, 0.0)
        return tf.reduce_mean(loss)
        
    def compute_bbox_loss(self, anchors, anchor_labels, anchor_mappings, gt_boxes, bbox_logits):
        indices = tf.where(tf.equal(anchor_labels, 1))[:, 0]  # only positive anchors

        anchors = tf.gather(anchors, indices)
        bbox_logits = tf.gather(bbox_logits, indices)
        bbox = utils.apply_anchor_deltas(anchors, bbox_logits)

        target_bbox = tf.gather(gt_boxes, anchor_mappings)
        target_bbox = tf.gather(target_bbox, indices)

        loss = self.smooth_l1_loss(target_bbox, bbox)
        loss = tf.maximum(loss, 0.0)
        loss = tf.abs(target_bbox - bbox)
        return tf.reduce_mean(loss)
        
    def smooth_l1_loss(self, y_true, y_pred):
        """
            y_true, y_pred: 2-D vector, [N, 4]
        """
        diff = tf.abs(y_true - y_pred)
        less_than_one = tf.cast(diff < 1.0, tf.float32)
        loss = (less_than_one * 0.5 * diff**2) + (1.0 - less_than_one) * (diff - 0.5)
        return loss
    
    
class FPN(object):
    def __init__(self, inp):
        layers = {'inp': inp}
        layers = self.build_conv_base(layers)
        layers = self.build_pyramid(layers)
        self.layers = layers

    def build_conv_base(self, layers):  # VGG-16
        with tf.variable_scope('vgg_16'):
            net = slim.repeat(layers['inp'], 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            layers['C1'] = net

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            layers['C2'] = net

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            layers['C3'] = net

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            layers['C4'] = net

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            layers['C5'] = net
        return layers

    def build_pyramid(self, layers):
        layers['P5'] = slim.conv2d(layers['C5'], 256, [1, 1], scope='p5')
        # layers['P6'] = slim.max_pool2d(layers['P5'], [2, 2], stride=2, scope='p6')

        for i in range(4, 1, -1):
            p, c = layers[f'P{i+1}'], layers[f'C{i}']
            upsample_shape = tf.shape(c)
            upsample = tf.image.resize_nearest_neighbor(p, [upsample_shape[1], upsample_shape[2]],
                                                        name=f'P{i}_upsample')
            c = slim.conv2d(c, 256, [1, 1], scope=f'P{i}_reduce_dimension')
            p = upsample + c
            p = slim.conv2d(p, 256, [3, 3], scope=f'P{i}')
            layers[f'P{i}'] = p
        return layers


if __name__ == '__main__':
    RPN()
