# Linear regression을 구현

import tensorflow as tf
import numpy as np

# 데이터
x_train = [1,2,3]
y_train = [1,2,3]

# Linear regression
# H(x) = Wx+b
W = tf.Variable(initial_value=tf.random_normal([1]), name = 'weight')
b = tf.Variable(initial_value=tf.random_normal([1]), name = 'bias')
h = x_train * W + b

#cost
cost = tf.reduce_mean(tf.square(h-y_train))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(2000):
    sess.run(train)
    if epoch % 20 == 0:
        print(epoch, sess.run(cost), sess.run(W), sess.run(b))

###################
tf.reset_default_graph()
print("########## tf.reset_default_graph()")

# 데이터
x_train = [1,2,3]
y_train = [1,2,3]

ones = np.ones(len(x_train))
print(x_train + ones)

print(len(x_train + ones))