# https://www.youtube.com/watch?v=oFGHOsAYiz0&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=28

import numpy as np
import tensorflow as tf
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)


X = tf.placeholder(dtype=tf.float32,shape=[None,2])
Y = tf.placeholder(dtype=tf.float32,shape=[None,1])
W = tf.Variable(tf.random_normal(shape=[2,1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32)

H = tf.sigmoid(tf.matmul(X,W)+b)
cost = -tf.reduce_mean(Y*tf.log(H)+(1-Y)*tf.log(1-H))
training = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
predict = tf.cast((H>0.5), dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y), tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(200):
    _, predic_val, acc_val = sess.run([training, predict, accuracy],feed_dict={X:x_data,Y:y_data})
    if epoch%100 ==0:
        print("Epoch: %s" %epoch)
        print("Prediction = %s" %predic_val)
        print("Accuracy = %s" %acc_val)

#Accuracy = 0.5 학습되지 않음
tf.reset_default_graph()

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)
X = tf.placeholder(dtype=tf.float32,shape=[None,2])
Y = tf.placeholder(dtype=tf.float32,shape=[None,1])

W1 = tf.Variable(tf.random_normal(shape=[2,2]), dtype=tf.float32)
b1 = tf.Variable(tf.random_normal(shape=[2]), dtype=tf.float32)
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal(shape=[2,1]), dtype=tf.float32)
b2 = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32)
H = tf.sigmoid(tf.matmul(layer1,W2)+b2)

cost = -tf.reduce_mean(Y*tf.log(H)+(1-Y)*tf.log(1-H))
training = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
predict = tf.cast((H>0.5), dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y), tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(200000):
    _, H_val, acc_val = sess.run([training, H, accuracy],feed_dict={X:x_data,Y:y_data})
    if epoch%100 ==0:
        print("Epoch: %s" %epoch)
        print("Prediction = %s" %H_val)
        print("Accuracy = %s" %acc_val)
