import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm, trange
from model_contrastive import Siamese

mnist = input_data.read_data_sets('data/', one_hot=False)

h, w, c = 28, 28, 1
batch_size = 128
learning_rate = 1e-4
num_iterations = 10_000
model_path = 'model/siamese/model'


def train():
    siamese = Siamese()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(siamese.loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in trange(num_iterations):
            x1, y1 = mnist.train.next_batch(batch_size)
            x2, y2 = mnist.train.next_batch(batch_size)

            x1 = np.reshape(x1, [-1, h, w, c])
            x2 = np.reshape(x2, [-1, h, w, c])

            y_ = (y1 == y2).astype(np.float32)
            feed_dict = {
                siamese.x1: x1,
                siamese.x2: x2,
                siamese.y_: y_,
                siamese.keep_prob: 0.5,
            }

            _, loss_v = sess.run([train_step, siamese.loss], feed_dict)
            assert not np.isnan(loss_v), 'Model diverged with loss = NaN'

            if i % 100 == 0:
                tqdm.write(f'step {i}: loss {loss_v}')

            if i % 1000 == 0:
                tqdm.write(f'Model saved: {saver.save(sess, model_path)}')

        print('Finished:', saver.save(sess, model_path))


if __name__ == '__main__':
    train()

