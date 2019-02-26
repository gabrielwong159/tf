import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from model import GAN
from tqdm import trange

mnist = input_data.read_data_sets('data/', reshape=False)

batch_size = 128
n_epochs = 20
model_path = 'model/gan/model'


def train():
    model = GAN()
    
    vars_d = slim.get_variables(scope='discriminator')
    d_optimizer = tf.train.RMSPropOptimizer(8e-4, decay=6e-8)
    d_train_step = d_optimizer.minimize(model.d_loss, var_list=vars_d)

    vars_g = slim.get_variables(scope='generator')
    g_optimizer = tf.train.RMSPropOptimizer(4e-4, decay=3e-8)
    g_train_step = g_optimizer.minimize(model.g_loss, var_list=vars_g)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in trange(n_epochs):
            writer = tf.summary.FileWriter(f'summaries/{epoch}', sess.graph)
            
            n_iters = int(np.ceil(len(mnist.train.labels) / batch_size))
            for i in trange(n_iters, desc=f'Epoch {epoch}', leave=False):
                x, _ = mnist.train.next_batch(batch_size)
                z = np.random.uniform(-1, 1, size=[batch_size, 100])

                _, _, summary = sess.run([d_train_step, g_train_step, model.summary], feed_dict={
                    model.x: x,
                    model.z: z,
                    model.keep_prob: 0.5,
                })
                writer.add_summary(summary, epoch*n_iters + i)
            saver.save(sess, model_path)
            

if __name__ == '__main__':
    train()
