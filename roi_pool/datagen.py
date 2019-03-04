import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import config

mnist = input_data.read_data_sets('data/', reshape=False)


def gen_image(width=128, height=128):
    x = np.random.randint(width - 28)
    y = np.random.randint(height - 28)

    image = np.zeros([height, width, 1], np.float32)
    image[y:y+28, x:x+28] = mnist.train.next_batch(1)[0]

    norm_x = (x + 14) / width
    norm_y = (y + 14) / height
    return image, (norm_x, norm_y)


def gen_batch(batch_size=32, fixed_size=True):
    images, labels = [], []
    for i in range(batch_size):
        if fixed_size:
            image, label = gen_image()
        else:
            h, w = config.h, config.w
            image, label = gen_image(width=w, height=h)
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

