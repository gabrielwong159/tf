import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm, trange
from model import FCN

vgg_model_path = 'model/vgg_16.ckpt'
fcn_model_path = 'model/fcn/model'
n_classes = 2

learning_rate = 1e-5
num_iterations = 10_000
batch_size = 1


def train():
    with tf.Session() as sess:
        # create and initialize all FCN variables
        fcn = FCN(n_classes)
        sess.run(tf.global_variables_initializer())
        # replace VGG net variables with pretrained weights
        restorer = tf.train.Saver(slim.get_variables_to_restore(include=['vgg_16']))
        restorer.restore(sess, vgg_model_path)
        # create optimizer only after weight restoration,
        # otherwise will try to restore optimizer variables for VGG
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(fcn.loss)
        sess.run(tf.variables_initializer(optimizer.variables()))
        
        saver = tf.train.Saver()
        
        for i in trange(num_iterations):
            inputs, masks = dataset.next_batch(batch_size)
            # inputs: array of images, shape [batch, h, w, 3], values [0 - 255]
            # masks: array of masks, shape [batch, h, w], values [0 - n_classes)
            _, loss = sess.run([train_step, loss], feed_dict={
                fcn.inputs: inputs,
                fcn.masks: masks,
                fcn.keep_prob: 0.5,
            })
            tqdm.write(f'Loss: {loss}')
            
        saver.save(sess, fcn_model_path)
        

if __name__ == '__main__':
    train()
