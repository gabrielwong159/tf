# understanding fpn - https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
# tf implementation - https://github.com/yangxue0827/FPN_Tensorflow/blob/master/libs/rpn/build_rpn.py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils


class RPN(object):
    POS_ANCHOR_THRESH = 0.7
    NEG_ANCHOR_THRESH = 0.1

    h, w, c = 224, 224, 1

    anchor_scales = [64, 32, 16, 8]
    anchor_ratios = [0.5, 1., 2.]
    backbone_shapes = np.array([
        [7, 7],
        [14, 14],
        [28, 28],
        [56, 56],
    ])
    feature_strides = [32, 16, 8, 4]


    def __init__(self):
        self.images = tf.placeholder(tf.float32, [None, self.h, self.w, self.c])
        self.gt_boxes = tf.placeholder(tf.float32, [None, None, 4])  # boxes should be normalized

        fpn = FPN(self.images)
        cls_logits_all, bbox_logits_all = self.build_rpn(fpn.layers)  # [batch, n_anchors, 2], [batch, n_anchors, 4]
        
        anchors = utils.generate_all_anchors(self.anchor_scales, self.anchor_ratios,
                                                  self.backbone_shapes, self.feature_strides,
                                                  anchor_stride=1, image_shape=[self.h, self.w])
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        
        cls_losses, bbox_losses = tf.map_fn(self.compute_loss, [self.gt_boxes, cls_logits_all, bbox_logits_all],
                                            dtype=(tf.float32, tf.float32))
        
        cls_loss = tf.reduce_mean(cls_losses)
        bbox_loss = tf.reduce_mean(bbox_losses)
        self.loss = cls_loss + bbox_loss
        
        pred_anchors = utils.apply_anchor_deltas(anchors=self.anchors, deltas=bbox_logits_all[0])
        pred_scores = tf.reduce_max(tf.nn.softmax(cls_logits_all[0]), axis=-1)
        pred_indices = tf.image.non_max_suppression(pred_anchors, pred_scores, iou_threshold=0.3, max_output_size=100)
        self.inference = tf.gather(pred_anchors, pred_indices, axis=0) * 224
        
        with tf.name_scope('summaries'):
            tf.summary.scalar('cls_loss', cls_loss)
            tf.summary.scalar('bbox_loss', bbox_loss)
            tf.summary.scalar('total_loss', self.loss)
        self.summaries = tf.summary.merge_all()
        
        # remove gt_boxes used for padding
        gt_boxes = self.gt_boxes[0]
        is_valid = tf.reduce_any(tf.not_equal(gt_boxes, -1), axis=-1)
        indices = tf.where(is_valid)[0]
        gt_boxes = tf.gather(gt_boxes, indices)
        anchors, anchor_mappings, anchor_labels = self.subsample_anchors(self.anchors, gt_boxes, cls_logits_all[0])
        self.anchors = utils.denorm_boxes(anchors, [self.h, self.w])
        self.mappings = anchor_mappings
        self.labels = anchor_labels
        
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
        cls_logits_all, bbox_logits_all = zip(*layers_outputs)
        cls_logits_all = tf.concat(cls_logits_all, axis=1)
        bbox_logits_all = tf.concat(bbox_logits_all, axis=1)
        return cls_logits_all, bbox_logits_all
    
    def compute_loss(self, args):
        gt_boxes, cls_logits, bbox_logits = args  # [n_gt_boxes, 4], [n_anchors, 2], [n_anchors, 4]
        # remove gt_boxes used for padding
        is_valid = tf.reduce_any(tf.not_equal(gt_boxes, -1), axis=-1)
        indices = tf.where(is_valid)[0]
        gt_boxes = tf.gather(gt_boxes, indices)
        
        # [6000, 4], [6000], [6000]
        anchors, anchor_mappings, anchor_labels = self.subsample_anchors(self.anchors, gt_boxes, cls_logits)
        cls_loss = self.compute_class_loss(anchor_labels, cls_logits)
        bbox_loss = self.compute_bbox_loss(anchors, anchor_labels, anchor_mappings, gt_boxes, bbox_logits)
        return cls_loss, bbox_loss
        
    def subsample_anchors(self, anchors, gt_boxes, cls_logits):
        # take top-n anchors by scores
        anchor_scores = cls_logits[:, 1]  # [n_anchors]
        pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[0])
        top_k_idx = tf.nn.top_k(anchor_scores, k=pre_nms_limit).indices
        anchors = tf.gather(anchors, top_k_idx)
        anchor_scores = tf.gather(anchor_scores, top_k_idx)

        # compute ious for anchors against given ground truth boxes
        ious = utils.compute_ious(anchors, gt_boxes)  # [n_anchors, n_gt_boxes]
        max_iou_per_anchor = tf.reduce_max(ious, axis=1)
        max_iou_per_gt_box = tf.reduce_max(ious, axis=0)

        labels = tf.zeros(shape=[tf.shape(anchors)[0]], dtype=tf.float32)
        mappings = tf.argmax(ious, axis=1)

        pos1 = max_iou_per_anchor >= self.POS_ANCHOR_THRESH
        pos2 = tf.reduce_sum(tf.cast(tf.equal(ious, max_iou_per_gt_box), tf.float32), axis=1)  # anchors with largest iou per gt box
        pos = pos1 | tf.cast(pos2, tf.bool)
        labels += tf.cast(pos, tf.float32)

        neg = max_iou_per_anchor < self.NEG_ANCHOR_THRESH
        labels -= tf.cast(neg, tf.float32)
        return anchors, mappings, labels

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
        indices = tf.where(tf.equal(anchor_labels, 1))  # only positive anchors

        anchors = tf.gather_nd(anchors, indices)
        bbox_logits = tf.gather_nd(bbox_logits, indices)
        bbox = utils.apply_anchor_deltas(anchors, bbox_logits)

        target_bbox = tf.map_fn(lambda idx: gt_boxes[idx], anchor_mappings, dtype=tf.float32)
        target_bbox = tf.gather_nd(target_bbox, indices)

        loss = self.smooth_l1_loss(target_bbox, bbox)
        loss = tf.maximum(loss, 0.0)
        return tf.reduce_mean(loss)
        
    def smooth_l1_loss(self, y_true, y_pred):
        """
            y_true, y_pred: 2-D vector, [N, 4]
        """
        diff = tf.abs(y_true - y_pred)
        less_than_one = tf.cast(diff < 1.0, tf.float32)
        loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
        return loss
    
    
class FPN(object):
    def __init__(self, inp):
        layers = {'inp': inp}
        layers = self.build_conv_base(layers)
        layers = self.build_pyramid(layers)
        self.layers = layers

    def build_conv_base(self, layers):  # VGG-16
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
