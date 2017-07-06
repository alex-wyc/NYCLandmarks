import tensorflow as tf
from PIL import Image
import PIL
import os
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import numpy as np


# utils
def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.Constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def read_image(path):
    img = Image.open(path)
    w,h = img.size
    pixels = img.load()
    output = [0 for _ in xrange(w * h)]
    for x in xrange(w):
        for y in xrange(h):
            output[y * w + x] = pixels[x, y]
    return np.asarray(output, dtype=np.int32)

def get_train_data(building):
    """
    return a tensor of the first 25 images in the numbered directory
    """
    img_subdir = "./dataset_processed/" + str(building) + "/"
    data = [read_image(img_subdir + str(x) + ".png") for x in xrange(26)]
    return np.asarray(data)

def get_test_data(building):
    """
    return the last 5 imgs in the numbered directory
    """
    img_subdir = "./dataset_processed/" + str(building) + "/"
    data = [read_image(img_subdir + str(x) + ".png") for x in xrange(26, 31)]
    return np.asarray(data)

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features, [-1, 600, 600, 1])

    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=100,
            kernel_size=[10,10],
            padding="same",
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 600 * 600 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=mode == learn.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=2)

    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return model_fn_lib.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# setup

x = tf.placeholder(tf.float32, [None, 360000])
y = tf.placeholder(tf.float32, [None, 2])

W_conv1 = weight_var([5, 5, 1, 32])
b_conv1 = bias_var([32])

x_image = tf.reshape(x, [-1, 600, 600, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W = tf.Variable(tf.zeros([360000, 10]))
b = tf.Variable(tf.zeros([10]))

W_conv2 = weight_var([5, 5, 32, 64])
b_conv2 = bias_var([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_var([7 * 7 * 64, 1024])
b_fc1 = bias_var([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_var([1024, 10])
b_fc2 = bias_var([10])

y_hat = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices = [1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

photos_dir = "./dataset_processed/"

for subdir in os.listdir(photos_dir):
    for file in os.listdir(photos_dir + subdir):

#for i in xrange(total):
#    batch = mnist.train.next_batch(per_batch)
#
#    if i % print_interval == 0:
#        train_accuracy = sess.run(accuracy, feed_dict = {
#            x: batch[0],
#            y: batch[1]
#            })
#        print("step %d, accuracy: %g" % (i, train_accuracy))
#
#    sess.run(train_step, feed_dict = {x: batch[0], y: batch[1]})
#
#print("test accuracy %g"%sess.run(accuracy, feed_dict={
#    x: mnist.test.images, y: mnist.test.labels}))
