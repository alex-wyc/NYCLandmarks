from __future__ import absolute_import, division, print_function

import tensorflow as tf
import random
import numpy as np
from PIL import Image

model_dir = "./model/"

WIDTH = 100
HEIGHT = 100
BATCH_SIZE = 20

def read_image(path):
    img = Image.open(path)
    w,h = img.size
    pixels = img.load()
    output = [0 for _ in xrange(w * h)]
    for x in xrange(w):
        for y in xrange(h):
            output[y * w + x] = pixels[x, y]
    return output

def get_train_data(building):
    """
    return a tensor of 25 imgs of the building and 30 of not the building
    """
    img_subdir = "./dataset_processed/" + str(building) + "/"
    data = [read_image(img_subdir + str(x) + ".png") for x in
            xrange(int(BATCH_SIZE / 2))]

    for i in xrange(int(BATCH_SIZE / 2)):
        file = random.randrange(0, 62 * 30)
        while file // 30 == building:
            file = random.randrange(0, 62 * 30)
        img = "./dataset_processed/" + str(file // 30) + "/" + str(file % 30) + ".png"
        try:
            data.append(read_image(img))
        except IOError:
            data.append(read_image("./dataset_processed/48/0.png"))

    return np.asarray(data)

def get_test_data(building):
    """
    return the last 5 imgs in the numbered directory and 5 of not the building
    """
    img_subdir = "./dataset_processed/" + str(building) + "/"
    data = [read_image(img_subdir + str(x) + ".png") for x in xrange(25)]

    for i in xrange(330):
        file = random.randrange(0, 62 * 30)
        img = "./dataset_processed/" + str(file // 30) + "/" + str(file % 30) + ".png"
        try:
            data.append(read_image(img))
        except IOError:
            data.append(read_image("./dataset_processed/48/0.png"))

    return np.asarray(data)

def get_test_labels(building):
    return np.asarray([[1, 0] for _ in xrange(25)] + [[0, 1] for _ in xrange(30)],
            dtype=np.int32)

def get_train_labels(building):
    return np.asarray([[1, 0] for _ in xrange(int(BATCH_SIZE / 2))] +
            [[0, 1] for _ in xrange(int(BATCH_SIZE / 2))], dtype=np.int32)



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def deep_cnn(input):
    with tf.name_scope("reshape"):
        x_img = tf.reshape(input, [-1, WIDTH, HEIGHT, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_var([5, 5, 1, 32])
        b_conv1 = bias_var([32])
        h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_var([5, 5, 32, 64])
        b_conv2 = bias_var([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('dense1'):
        W_dense1 = weight_var([WIDTH * HEIGHT * 4, 1024])
        b_dense1 = bias_var([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, WIDTH * HEIGHT * 4])
        h_dense1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_dense1) + b_dense1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_dense_drop = tf.nn.dropout(h_dense1, keep_prob)

    with tf.name_scope('dense2'):
        W_dense2 = weight_var([1024, 2])
        b_dense2 = bias_var([2])

        y_conv = tf.matmul(h_dense_drop, W_dense2) + b_dense2

    return y_conv, keep_prob

def train(building):
    x = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT])
    y_ = tf.placeholder(tf.float32, [None, 2])

    y_conv, keep_prob = deep_cnn(x)

    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = model_dir + str(building)
    print("Saving graph to: %s" % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(200):
            train_data = get_train_data(building)
            train_labels = get_train_labels(building)

            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: train_data, y_: train_labels, keep_prob: 1.0
                })
                print("step %d, training accuracy %g" % (i, train_accuracy))

            train_step.run(feed_dict = {
                x: train_data, y_: train_labels, keep_prob: 0.5
            })

        test_data = get_test_data(building)
        test_labels = get_test_labels(building)

        print("test accuracy %g" % accuracy.eval(feed_dict = {
            x: test_data, y_:test_labels, keep_prob: 1.0
        }))

train(0)
