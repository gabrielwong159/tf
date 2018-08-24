import numpy as np
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(1)
tf.set_random_seed(1)

mnist = input_data.read_data_sets('data/', one_hot=False)


def make_crop(image):
    scale = np.random.uniform(0.5, 1.5, size=[2])
    w, h = (scale * 28).astype(np.int64)
    x = np.random.randint(image.shape[1] - w)
    y = np.random.randint(image.shape[0] - h)
    return y, x, y+h, x+w


def add_crop(image, train):
    if train:
        crop, label = mnist.train.next_batch(1)
    else:
        crop, label = mnist.test.next_batch(1)
    crop = crop.reshape([28, 28])

    y1, x1, y2, x2 = make_crop(image)
    crop = cv2.resize(crop, (x2 - x1, y2-y1))

    image[y1:y2, x1:x2] = crop
    return int(label), [y1, x1, y2, x2]


def generate_image(train=True):
    image = np.zeros([224, 224], np.float32)
    n_crops = np.random.randint(1, 5)
    gt_cls, gt_boxes = zip(*[add_crop(image, train) for i in range(n_crops)])
    gt_cls, gt_boxes = map(np.array, (gt_cls, gt_boxes))
    return image, gt_cls, gt_boxes
