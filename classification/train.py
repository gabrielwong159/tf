import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # deprecated
from model import MNIST
from tqdm import tqdm, trange

learning_rate = 1e-4
num_iterations = 50_00
batch_size = 50
model_path = 'model/mnist/model'

mnist = input_data.read_data_sets('data/', one_hot=False)


def train():
    model = MNIST()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in trange(num_iterations):
            x, y = mnist.train.next_batch(batch_size)
            x = x.reshape([-1, model.h, model.w, model.c])

            if i % 100 == 0:
                train_accuracy = sess.run(model.accuracy, feed_dict={
                    model.x: x,
                    model.y: y,
                    model.keep_prob: 1.0,
                })
                tqdm.write(f'Step {i}, training accuracy: {train_accuracy}')

            sess.run(train_step, feed_dict={
                model.x: x,
                model.y: y,
                model.keep_prob: 0.5,
            })

        print('Training completed, model saved at:', saver.save(sess, model_path))


if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    train()

