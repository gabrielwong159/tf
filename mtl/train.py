import tensorflow as tf
from tqdm import tqdm, trange
import data_loader as data
from model import MNIST

batch_size = 50
learning_rate = 1e-4
num_iterations = 20_000
model_path = 'model/model'


def train():
    model = MNIST()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in trange(num_iterations):
            images, class_labels, color_labels = data.next_batch(True, batch_size)
            sess.run(train_step, feed_dict={
                model.x: images,
                model.y_class: class_labels,
                model.y_color: color_labels,
                model.keep_prob: 0.5,
            })

        print(saver.save(sess, model_path))


if __name__ == '__main__':
    train()

