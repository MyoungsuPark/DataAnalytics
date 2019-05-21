# https://www.youtube.com/watch?v=oSJfejG2C3w&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=20

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]


import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,3])
W = tf.Variable(tf.random_normal(shape=[3,3]))
b = tf.Variable(tf.random_normal(shape=[3]))


#hypo = tf.nn.softmax (tf.matmul(X,W)+b)
#cost = cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypo), axis=1))
logits = tf.matmul(X,W)+b
hypo = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predict = tf.arg_max(hypo, dimension=1)
Y_index = tf.arg_max(Y, dimension=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y_index),dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(200000):
    cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
    if epoch % 200 == 0:
        print("Epoch:%s Cost_val: %s" %(epoch,cost_val))

acc_val = sess.run(accuracy, feed_dict={X:x_test, Y:y_test})
print("Acc_val:%s" %acc_val)




