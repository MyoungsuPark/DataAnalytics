import tensorflow as tf

hello = tf.constant("Hello Tensorflow")
sess = tf.Session()
print(sess.run(hello))
# b 문자열이 출력될 것이다. byte literal을 의미한다.


# computational Graph
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = node1+node2
print("node1 = %s" %node1)
print("node3 = %s" %node3)

sess = tf.Session()
print("sess.run(node3) = %d" %sess.run(node3))
print("sess.run(node2) = %d" %sess.run(node2))


# Placeholder
tf.reset_default_graph()
node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
# 아래처럼 했을때 차이를 느껴보시라.
# node1 = tf.placeholder(tf.float32, shape=[None])
# node2 = tf.placeholder(tf.float32, shape=[None])

node3 = tf.add(node1, node2)
node4 = node1+node2

with tf.Session() as sess:
    print("node4.eval() = %s" %node4.eval(feed_dict={node1:3.0, node2:4.0}))
    print("node4.eval() = %s" % node4.eval(feed_dict={node1: [3.0, 4.0], node2: 4.0}))
    print("node4.eval() = %s" % node4.eval(feed_dict={node1: [3.0, 4.0], node2: [4.0,5]}))

sess = tf.Session()
print("sesion.run(node4) = %s" %sess.run(node4, feed_dict={node1: [3.0, 4.0], node2: [4.0,5]}))
