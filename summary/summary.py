import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from model import MNIST
from os.path import join
from tqdm import trange

summaries_dir = 'summaries'
learning_rate = 1e-4
batch_size = 50
num_iterations = 5_000

mnist = input_data.read_data_sets('data/', one_hot=False)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
        
def make_summaries(model):
    for layer in ['conv1', 'conv2', 'fc1', 'fc2']:
        for var_type in ['weights', 'biases']:
            with tf.name_scope(layer), tf.name_scope(var_type):
                var = '/'.join([layer, var_type])
                variable_summaries(slim.get_variables_by_name(var)[0])
    tf.summary.histogram('keep_prob', model.keep_prob)
    tf.summary.histogram('predictions', model.logits)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summaries = tf.summary.merge_all()
    return merged_summaries
        
        
def main():
    tf.reset_default_graph()
    model = MNIST()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    
    with tf.Session() as sess:
        merged_summaries = make_summaries(model)
        train_writer = tf.summary.FileWriter(join(summaries_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(join(summaries_dir, 'test'))
        
        sess.run(tf.global_variables_initializer())
        
        for i in trange(num_iterations):
            feed_shape = [-1, model.h, model.w, model.c]
            if i % 10 == 0:
                summary = sess.run(merged_summaries, feed_dict={
                    model.x: mnist.test.images.reshape(feed_shape),
                    model.y: mnist.test.labels,
                    model.keep_prob: 1.0,
                })
                test_writer.add_summary(summary, i)
            else:
                x, y = mnist.train.next_batch(batch_size)
                _, summary = sess.run([train_step, merged_summaries], feed_dict={
                    model.x: x.reshape(feed_shape),
                    model.y: y,
                    model.keep_prob: 0.5,
                })
                train_writer.add_summary(summary, i)

                
if __name__ == '__main__':
    main()
