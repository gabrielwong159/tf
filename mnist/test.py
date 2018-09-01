import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import MNIST

batch_size = 1024
model_path = 'model/mnist/model'

mnist = input_data.read_data_sets('data/', one_hot=False, reshape=False)


def test():
    model = MNIST()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        test_accuracy = sess.run(model.accuracy, feed_dict={
            model.x: mnist.test.images,
            model.y: mnist.test.labels,
            model.keep_prob: 1.0,
        })
    print('Test accuracy:', test_accuracy)


if __name__ == '__main__':
    test()

