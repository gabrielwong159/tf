import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm, trange
from model import VAE

mnist = input_data.read_data_sets('data/')

batch_size = 64
learning_rate = 5e-4
num_iterations = 10_000
model_path = 'model/vae/model'


def train():
    model = VAE(batch_size)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        for i in trange(num_iterations):
            sess.run(tf.global_variables_initializer())

            x, y = mnist.train.next_batch(batch_size)
            _, loss = sess.run([train_step, model.loss], feed_dict={
                model.x: x.reshape([-1, 28, 28, 1]),
            })

            if i % 10 == 0:
                tqdm.write(f'Step {i}, loss: {loss}')
            if i % 100 == 0:
                saver.save(sess, model_path)
    
    print(saver.save(sess, model_path))

    
if __name__ == '__main__':
    train()
