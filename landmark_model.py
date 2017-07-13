import tensorflow as tf
import random
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
    return np.asarray(output, dtype=np.float32)

def get_train_data(building):
    """
    return a tensor of 25 imgs of the building and 30 of not the building
    """
    img_subdir = "./dataset_processed/" + str(building) + "/"
    data = [read_image(img_subdir + str(x) + ".png") for x in xrange(25)]

    for i in xrange(30):
        try:
            file = random.randrange(0, 62 * 30)
            img = "./dataset_processed/" + str(file / 30) + "/" + str(file % 30) + ".png"
            data.append(read_image(img))
        except:
            i-=1

    return np.asarray(data)

def get_test_data(building):
    """
    return the last 5 imgs in the numbered directory and 5 of not the building
    """
    img_subdir = "./dataset_processed/" + str(building) + "/"
    data = [read_image(img_subdir + str(x) + ".png") for x in xrange(26, 31)]

    for i in xrange(5):
        file = random.randrange(0, 62 * 30)
        img = "./dataset_processed/" + str(file / 30) + "/" + str(file % 30) + ".png"
        data.append(read_image(img))

    return np.asarray(data)

def get_test_labels(building):
    return np.asarray([1 for _ in xrange(25)] + [0 for _ in xrange(30)])

def get_train_labels(building):
    return np.asarray([1 for _ in xrange(5)] + [0 for _ in xrange(5)])

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features, [-1, 50, 50, 1])

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

    pool2_flat = tf.reshape(pool2, [-1, 50 * 50 * 64])
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

build_building_classifier = lambda x: learn.Estimator(
        model_fn=cnn_model_fn, model_dir="./models/" + str(x))

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
        every_n_iter=5)

metrics = {
    "accuracy":
        learn.MetricSpec(
            metric_fn=tf.metrics.accuracy, prediction_key="classes"),
}

def train(x):
    classifier = build_building_classifier(x)
    classifier.fit(
        x=get_train_data(x),
        y=get_train_labels(x),
        batch_size=5,
        steps=10,
        monitors=[logging_hook])
    results = classifier.evaluate(
        x=get_test_data(x), y=get_test_labels(x), metrics=metrics)
    print(results)

train(0)

