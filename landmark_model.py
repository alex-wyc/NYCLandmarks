import random
from PIL import Image
import PIL
import os
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import numpy as np

BATCH_SIZE = 10


# utils
def read_image(path):
    img = Image.open(path)
    w,h = img.size
    pixels = img.load()
    output = [0 for _ in xrange(w * h)]
    for x in xrange(w):
        for y in xrange(h):
            output[y * w + x] = (float(pixels[x, y]) - 127.5) / 255.
    return np.asarray(output, dtype=np.float32)

def get_train_data(building):
    """
    return a tensor of 25 imgs of the building and 30 of not the building
    """
    img_subdir = "./dataset_processed/" + str(building) + "/"
    data = [read_image(img_subdir + str(x) + ".png") for x in xrange(BATCH_SIZE / 2)]

    for i in xrange(BATCH_SIZE / 2):
        file = random.randrange(0, 62 * 30)
        while file / 30 == building:
            file = random.randrange(0, 62 * 30)
        img = "./dataset_processed/" + str(file / 30) + "/" + str(file % 30) + ".png"
        data.append(read_image(img))

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
    return np.asarray([1 for _ in xrange(25)] + [0 for _ in xrange(30)],
            dtype=np.int32)

def get_train_labels(building):
    return np.asarray([1 for _ in xrange(BATCH_SIZE / 2)] + [0 for _ in
        xrange(BATCH_SIZE / 2)], dtype=np.int32)

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 100, 100, 1])

    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[10,10],
            padding="same",
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # output tensor size [-1, 50, 50, 32]

    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # output tensor size [-1, 25, 25, 64]

    pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=(logits + tf.constant(1e-8)))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }


    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

build_building_classifier = lambda x: tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./models/" + str(x))

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
        every_n_iter=1)

metrics = {
    "accuracy":
        learn.MetricSpec(
            metric_fn=tf.metrics.accuracy, prediction_key="classes"),
}

def train(x):
    classifier = build_building_classifier(x)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":get_train_data(x)},
            y=get_train_labels(x),
            batch_size=BATCH_SIZE,
            num_epochs = None,
            shuffle=True)

    classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": get_test_data(x)},
            y=get_test_labels(x),
            num_epochs=1,
            shuffle=False)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

def main(unused_argv):
    train(0)

if __name__ == "__main__":
    tf.app.run()


#mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#train_data = mnist.train.images # Returns np.array
#train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#eval_data = mnist.test.images # Returns np.array
#eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#print train_data.shape
#print train_labels.shape
#print eval_data.shape
#print eval_labels.shape
#
#print get_train_data(0).shape
#print get_train_labels(0).shape
