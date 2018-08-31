import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

x_data = np.linspace(-1, 1, 100)
y_data = 0.8*x_data + 0.4

Weights = tf.Variable(tf.zeros([1]))
bias = tf.Variable(tf.zeros([1]))

y = Weights*x_data+bias
loss = tf.reduce_mean(tf.square(y-y_data))

opt = tf.train.GradientDescentOptimizer(0.1)
train = opt.minimize(loss)


#initialize all var
init = tf.global_variables_initializer()

#create session
sess = tf.Session()
sess.run(init) #key

for epoch in range(200):
    sess.run(train)
    if epoch%10 == 0:
        print sess.run(loss), sess.run(Weights), sess.run(bias)