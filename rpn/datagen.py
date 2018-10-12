import numpy as np
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils
from model import RPN

np.random.seed(1)
tf.set_random_seed(1)

dtype = np.float32
mnist = input_data.read_data_sets('data/', one_hot=False, dtype=dtype)
max_crops = 4


def make_crop(image):
    scale = np.random.uniform(0.5, 1.5, size=[2])
    w, h = (scale * 28).astype(np.int64)
    x = np.random.randint(image.shape[1] - w)
    y = np.random.randint(image.shape[0] - h)
    return y, x, y+h, x+w


def add_crop(image, train):
    """
    Return:
        label: int value indicating image class
        gt_box: Un-normalized bounding box coordinates in [y1, x1, y2, x2]
    """
    if train:
        crop, label = mnist.train.next_batch(1)
    else:
        crop, label = mnist.test.next_batch(1)
    crop = crop.reshape(28, 28)

    y1, x1, y2, x2 = make_crop(image)
    crop = cv2.resize(crop, (x2 - x1, y2-y1))

    image[y1:y2, x1:x2] = cv2.bitwise_or(image[y1:y2, x1:x2], crop)
    return int(label), [y1, x1, y2, x2]


def generate_image(train):
    """
    Return:
        image: Image as np.ndarray
        gt_cls: Array of classes of crops in [N,]
        gt_boxes: Array of normalized bounding boxes for each crop in [N, (y1, x2, y2, x2)]
    """
    image = np.zeros([RPN.h, RPN.w], dtype=dtype)
    
    n_crops = np.random.randint(1, max_crops + 1)
    gt_cls, gt_boxes = map(np.array, zip(*[add_crop(image, train) for i in range(n_crops)]))
    
    padding_boxes = -np.ones([max_crops - n_crops, 4], np.float64)
    gt_boxes = np.concatenate([gt_boxes, padding_boxes], axis=0)
    gt_boxes = utils.norm_boxes(gt_boxes, [RPN.h, RPN.w])
    
    image = cv2.merge([image] * 3)
    return image, gt_cls, gt_boxes


def generate_batch(batch_size, train=True):
    batch = zip(*[generate_image(train) for _ in range(batch_size)])
    images, gt_cls, gt_boxes = map(np.array, batch)
    
    images = images.reshape(-1, RPN.h, RPN.w, RPN.c)
    return images, gt_cls, gt_boxes
