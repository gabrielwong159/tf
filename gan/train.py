import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from model import GAN
from tqdm import tqdm, trange

mnist = input_data.read_data_sets('data/', one_hot=True)

g_lr = 2e-4
d_lr = 1e-4
batch_size = 128
n_epochs = 20
model_path = 'model/gan/model'


def train():
    model = GAN()
    g_train_step = tf.train.AdamOptimizer(g_lr).minimize(model.g_loss, var_list=slim.get_variables(scope='generator'))
    d_train_step = tf.train.AdamOptimizer(d_lr).minimize(model.d_loss, var_list=slim.get_variables(scope='discriminator'))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)

        for epoch in range(n_epochs):
            summary_writer = tf.summary.FileWriter(f'summaries/{epoch}', sess.graph)
            n_iters = int(np.ceil(len(mnist.train.images) / batch_size))
            for i in trange(n_iters, desc=f'Epoch {epoch}'):
                x, y = mnist.train.next_batch(batch_size)
                x = x.reshape([-1, 28, 28, 1])
                x = x*2 - 1
                x += np.random.normal(-1, 1, size=x.shape)

                z = np.random.normal(-1, 1, size=[batch_size, 100])

                _, _, summary = sess.run([g_train_step, d_train_step, model.summary], feed_dict={
                    model.x: x,
                    model.z: z,
                    model.keep_prob: 0.5,
                })
                summary_writer.add_summary(summary, epoch*n_iters + i)
            saver.save(sess, model_path)


if __name__ == '__main__':
    train()

