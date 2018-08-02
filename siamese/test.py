import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange
from model import Siamese

mnist = input_data.read_data_sets('data/', one_hot=False)

h, w, c = 28, 28, 1
batch_size = 128
model_path = 'model/siamese/model'


def test():
    siamese = Siamese()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        x1, y1 = mnist.test.next_batch(batch_size)
        x2, y2 = mnist.test.next_batch(batch_size)

        x1 = np.reshape(x1, [-1, h, w, c])
        x2 = np.reshape(x2, [-1, h, w, c])

        y_true = (y1 == y2).astype(np.float32)

        out = sess.run(siamese.out, feed_dict={
            siamese.x1: x1,
            siamese.x2: x2,
            siamese.keep_prob: 1.0,
        })
        y_pred = np.argmax(out, axis=-1)

    print(np.sum(y_pred == y_true))


if __name__ == '__main__':
    test()

