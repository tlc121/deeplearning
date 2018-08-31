#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def compute_accuracy(x, y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_in:x, keep_prob:1})
    correct = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    res = sess.run(accuracy, feed_dict={x_in:x, y_in:y, keep_prob:1})
    return res

def add_layer(inputs, input_size, output_size, activation_function = None):
    #define layer name
    with tf.name_scope('layer'):
        #define the weights name
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([input_size, output_size]), name='W')
        #define biases
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, output_size]), name='b')
        with tf.name_scope('output'):
            res = tf.matmul(inputs, Weights) + bias
            res = tf.nn.dropout(res, keep_prob)
        if activation_function is None:
            return res
        else:
            return activation_function(res)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#设置输入和输出变量
x_in = tf.placeholder(tf.float32, [None, 784])
y_in = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#搭建网络
prediction = add_layer(x_in, 784, 10, tf.nn.softmax)

#构建损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(prediction)*y_in, reduction_indices=[1]))

#构建训练的优化器
train = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#初始化
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(10000):
    batch_x, batch_y = mnist.train.next_batch(32)
    sess.run(train, feed_dict={x_in:batch_x, y_in:batch_y, keep_prob:1})
    if epoch%1000 == 0:
        print compute_accuracy(mnist.test.images, mnist.test.labels)
