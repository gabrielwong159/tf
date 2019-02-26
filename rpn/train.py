import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm, trange
from datagen import generate_batch
from model import RPN

learning_rate = 1e-3
batch_size = 32
n_iterations = 20_000

model_desc = f'lr{learning_rate}_batch{batch_size}_iters{n_iterations}'
model_path = f'model/l1_loss/{model_desc}/model'
summaries_path = f'logs/l1_loss/{model_desc}'

def train():
    model = RPN()
    saver = tf.train.Saver()
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(summaries_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        
        for i in trange(n_iterations):
            images, gt_cls, gt_boxes = generate_batch(batch_size, train=True)
            _, loss, summaries = sess.run([train_step, model.loss, model.summaries], feed_dict={
                model.images: images,
                model.gt_boxes: gt_boxes,
            })
            summary_writer.add_summary(summaries, i)
            assert not np.isnan(loss), 'loss == NaN'
            
            if i % 1000 == 0:
                saver.save(sess, model_path)
        print(saver.save(sess, model_path))


if __name__ == '__main__':
    train()
