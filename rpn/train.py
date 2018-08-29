import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm, trange
from datagen import generate_batch
from model import RPN

learning_rate = 5e-4
momentum = 0.9
batch_size = 32
n_iterations = 70_000
model_path = 'model/rpn/model'


def train():
    model = RPN()
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps=50_000, decay_rate=0.1)
    optimizer = tf.train.MomentumOptimizer(lr, momentum)
    train_step = optimizer.minimize(model.loss, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('summaries', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        for i in trange(n_iterations):
            images, gt_cls, gt_boxes = generate_batch(batch_size, train=True)
            _, loss, summaries = sess.run([train_step, model.loss, model.summaries], feed_dict={
                model.images: images,
                model.gt_boxes: gt_boxes,
            })
            summary_writer.add_summary(summaries, i)
            assert not np.isnan(loss), 'loss == NaN'
            
            if i % 100 == 0:
                tqdm.write(f'step {i}: loss {loss}')
                saver.save(sess, model_path)
        print(saver.save(sess, model_path))


if __name__ == '__main__':
    train()
