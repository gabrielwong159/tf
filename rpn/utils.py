import numpy as np
import tensorflow as tf


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    Parameters:
        scales: 1-D array of anchor sizes in pixels, e.g. [8, 16, 32]
        ratios: 1-D array of aspect ratios of width/height, e.g. [0.5, 1., 2.]
        shape: Shape of the feature map in [h, w]
        feature_stride: Stride of the feature map relative to the image
        anchor_stride: Stride of the anchors (e.g. stride=2 means anchor on every other pixel)

    Returns:
        anchors - [N, [y1, x1, y2, x2]]
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    shifts_x = np.arange(shape[1], step=anchor_stride) * feature_stride
    shifts_y = np.arange(shape[0], step=anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    box_w, box_cx = np.meshgrid(widths, shifts_x)
    box_h, box_cy = np.meshgrid(heights, shifts_y)

    box_centers = np.stack([box_cy, box_cx], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_h, box_w], axis=2).reshape([-1, 2])
    # convert to (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5*box_sizes,
                            box_centers + 0.5*box_sizes], axis=1)
    return boxes


def generate_all_anchors(scales, ratios, backbone_shapes,
                         feature_strides, anchor_stride,
                         image_shape):
    """
    Given multiple backbone shapes and feature strides corresponding to layers,
    create anchors for each feature map.
    
    Parameters:
        scales: 1-D array of anchor sizes in pixels, e.g. [8, 16, 32]
        ratios: 1-D array of aspect ratios of width/height, e.g. [0.5, 1., 2.]
        backbone_shapes: Shape of the feature map in [h, w]
        feature_stride: Stride of the feature map relative to the image
        anchor_stride: Stride of the anchors (e.g. stride=2 means anchor on every other pixel)
        image_shape: Shape of original image in [h, w], used to normalize anchors
        
    Returns:
        anchors - [N, [y1, x1, y2, x2]]
    """
    anchors = [generate_anchors(scales, ratios, shape, stride, anchor_stride)
               for shape, stride in zip(backbone_shapes, feature_strides)]
    anchors = np.concatenate(anchors, axis=0)
    anchors = norm_boxes(anchors, image_shape)
    return anchors


def compute_ious(anchors, gt_boxes):
    """
        anchors: [N, 4] - (y1, x1, y2, x2)
        gt_boxes: [M, 4] - (y1, x1, y2, x2)
    """
    a_y1, a_x1, a_y2, a_x2 = tf.split(anchors, 4, axis=1)  # a_y1 - [N, 1]
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.unstack(gt_boxes, axis=1)  # gt_y1 - [M,]
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

    y1 = tf.maximum(a_y1, gt_y1)
    y2 = tf.minimum(a_y2, gt_y2)
    x1 = tf.maximum(a_x1, gt_x1)
    x2 = tf.minimum(a_x2, gt_x2)

    intersection = tf.maximum(0., x2 - x1) * tf.maximum(0., y2 - y1)
    union = a_area + gt_area - intersection
    return intersection / union


def apply_anchor_deltas(boxes, deltas):
    """
    Applies the given deltas to the given boxes.
    
    Parameters:
        anchors: [N, (y1, x1, y2, x2)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to (y, x, h, w)
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to (y1, x1, y2, x2)
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1)
    return result


def norm_boxes(boxes, shape):
    """
    Converts boxes from pixel coordinates to normalized coordinates.
    
    Parameters:
        boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        shape: [height, width] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h-1, w-1, h-1, w-1])
    shift = np.array([0, 0, 1, 1])
    return ((boxes - shift) / scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """
    Converts boxes from normalized coordinates to pixel coordinates.
    
    Parameters:
        boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        shape: [height, width] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)
