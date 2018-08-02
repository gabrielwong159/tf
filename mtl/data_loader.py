import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2

np.random.seed(0)

h, w = 28, 28
colors = [
    (255, 0, 0),
    (255, 127, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 0, 255),
    (75, 0, 130),
    (148, 0, 211)
]
n_colors = len(colors)
mnist = input_data.read_data_sets('data/', one_hot=False)


def next_batch(train, batch_size):
    dataset = mnist.train if train else mnist.test
    x, class_labels = dataset.next_batch(batch_size)
    x = np.reshape(x, [-1, h, w])

    color_labels = np.random.randint(n_colors, size=[batch_size])
    images = np.zeros([batch_size, h, w, 3])

    for i in range(batch_size):
        images[i][np.round(x[i]).astype(np.bool)] = colors[color_labels[i]]

    return images, class_labels, color_labels
