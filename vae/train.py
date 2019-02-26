import numpy as np
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm, trange
from model import VAE

mnist = input_data.read_data_sets('data/', reshape=False)

n_epochs = 16
batch_size = 64
learning_rate = 1e-4
model_path = 'model/vae_xent/model'


def train():
    model = VAE()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = mnist.train.images
        n_iters = int(np.ceil(len(x) / batch_size))
        
        for epoch in range(n_epochs):
            for i in trange(n_iters, desc=f'Epoch {epoch}'):
                _, loss = sess.run([train_step, model.loss], feed_dict={
                    model.x: x[i::n_iters],
                })

                if i % 100 == 0:
                    tqdm.write(f'Step {i}, loss: {loss}')

            print(saver.save(sess, model_path))
            
            test_image = mnist.test.images[:1]
            out = sess.run(model.out, feed_dict={model.x: test_image})
            out_image = np.squeeze(out[0]) * 255
            cv2.imwrite(f'samples/{epoch}_{int(loss)}.png', out_image)

    
if __name__ == '__main__':
    train()
