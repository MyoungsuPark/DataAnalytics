# https://www.youtube.com/watch?v=2FeWGgnyLSw&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=13

import tensorflow as tf

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X = tf.placeholder(tf.float32, shape=[None, 2], name='TrainingData')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='TargetValue')
W = tf.Variable(tf.random_normal(shape=[2, 1]), name='Weight')
b = tf.Variable(tf.random_normal(shape=[1]), name='Bias')

H = tf.sigmoid(tf.matmul(X,W)+b)
cost = -tf.reduce_mean(Y*tf.log(H)+(1-Y)*tf.log(1-H))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

predict = tf.cast((H>0.5), dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(10000):
    accu_val, cost_val,_ = sess.run([accuracy, cost, train], feed_dict={X:x_data, Y:y_data})
    if epoch%20==0:
        print("Accuracy=%s Cost=%s" %(accu_val,cost_val))

