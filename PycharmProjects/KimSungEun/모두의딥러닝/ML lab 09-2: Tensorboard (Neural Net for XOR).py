# https://www.youtube.com/watch?v=lmrWZPFYjHM&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=29


# Lab 9 XOR
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2], name="x")
Y = tf.placeholder(tf.float32, [None, 1], name="y")

with tf.name_scope("Layer1"):
    W1 = tf.Variable(tf.random_normal([2, 2]), name="weight_1")
    b1 = tf.Variable(tf.random_normal([2]), name="bias_1")
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    #### Tensorboard
    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Layer1", layer1)
    #### Tensorboard

with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random_normal([2, 1]), name="weight_2")
    b2 = tf.Variable(tf.random_normal([1]), name="bias_2")
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    #### Tensorboard
    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Hypothesis", hypothesis)
    #### Tensorboard

# cost/loss function
with tf.name_scope("Cost"):
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    #### Tensorboard
    tf.summary.scalar("Cost", cost)
    #### Tensorboard

with tf.name_scope("Train"):
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

#### Tensorboard
tf.summary.scalar("accuracy", accuracy)
#### Tensorboard

# Launch graph
with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs

    #### Tensorboard Step#2
    merged_summary = tf.summary.merge_all()
    #### Tensorboard Step#2

    #### Tensorboard Step#3. Create Writer and add graph
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
    writer.add_graph(sess.graph)  # Show the graph
    #### Tensorboard Step#3. Create Writer and add graph

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        #### Tensorboard Step#4.1 Run Summary merge
        _, summary, cost_val = sess.run(
            [train, merged_summary, cost], feed_dict={X: x_data, Y: y_data}
        )
        #### Tensorboard Step#4.1 Run Summary merge

        #### Tensorboard Step#4.2 add_summary
        writer.add_summary(summary, global_step=step)
        #### Tensorboard Step#4.2 add_summary


        if step % 100 == 0:
            print(step, cost_val)

    # Accuracy report
    h, p, a = sess.run(
        [hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data}
    )

    print(f"\nHypothesis:\n{h} \nPredicted:\n{p} \nAccuracy:\n{a}")