import tensorflow as tf

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

# setup

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 2])

W_conv1 = weight_var([5, 5, 1, 32])
b_conv1 = bias_var([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

W = tf.Variable(tf.zeros([784, 10]))
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

for i in xrange(total):
    batch = mnist.train.next_batch(per_batch)

    if i % print_interval == 0:
        train_accuracy = sess.run(accuracy, feed_dict = {
            x: batch[0],
            y: batch[1]
            })
        print("step %d, accuracy: %g" % (i, train_accuracy))

    sess.run(train_step, feed_dict = {x: batch[0], y: batch[1]})

print("test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y: mnist.test.labels}))
