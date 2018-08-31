#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
        if activation_function is None:
            return res
        else:
            return activation_function(res)

x_data = np.linspace(-1, 1, 100, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.01, x_data.shape).astype(np.float32)
y_data = x_data**2 + noise

#用name_scope来形成一个大图层，图层名字就是括号里的'inputs'
with tf.name_scope('inputs'):
    x_var = tf.placeholder(tf.float32, [None, 1], name='x_in')
    y_var = tf.placeholder(tf.float32, [None, 1], name='y_in')

l1 = add_layer(x_var, 1, 10, tf.nn.sigmoid)
l2 = add_layer(l1, 10, 10, tf.nn.sigmoid)
l3 = add_layer(l2, 10, 1, None)

#reduction_indices是将列维度进行求和
loss = tf.reduce_mean(tf.square(l3-y_var))
train = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
#让图像一直保留
plt.ion()
plt.show()

for epoch in range(3000):
    sess.run(train, feed_dict={x_var:x_data, y_var:y_data})
    if epoch%100 == 0:
        print sess.run(loss, feed_dict={x_var:x_data, y_var:y_data})
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction = sess.run(l3, feed_dict={x_var:x_data})
        lines = ax.plot(x_data, prediction, 'r-', lw=5)
        plt.pause(0.5)