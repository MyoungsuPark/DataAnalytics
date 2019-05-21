# https://www.youtube.com/watch?v=VRnubDzIy3A&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=16

# Lab 6 Softmax Classifier
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder(dtype=tf.float32, shape=[None,4])
Y = tf.placeholder(dtype=tf.float32, shape=[None,3])
W = tf.Variable(tf.random_normal(shape=[4,3]))
b = tf.Variable(tf.random_normal(shape=[3]))

hypo = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypo), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)

predict = tf.arg_max(hypo, dimension=1)
Y_index = tf.arg_max(Y, dimension=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y_index),dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(200000):
    predic_val, accu_val, cost_val, _ = sess.run([predict, accuracy, cost, train], feed_dict={X: x_data, Y:y_data})
    if epoch % 100==0:
        print("Epoch:%s. Cost:%s" %(epoch, cost_val))
        print("Accuracy= %s" %accu_val)



