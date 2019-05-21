# https://www.youtube.com/watch?v=6CCXyfvubvY&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=34
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None,784])
Y = tf.placeholder(tf.float32, shape=[None,10])
W = tf.Variable(tf.random_normal(shape=[784,10]))
b = tf.Variable(tf.random_normal(shape=[10]))

logits = tf.matmul(X,W)+b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
hypo = tf.nn.softmax(logits)

training = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(2000):
    avg_cost = 0
    to
