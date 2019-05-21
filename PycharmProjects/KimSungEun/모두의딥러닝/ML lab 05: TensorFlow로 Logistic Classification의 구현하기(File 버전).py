# https://www.youtube.com/watch?v=2FeWGgnyLSw&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=13

import tensorflow as tf

# x_data = [[1, 2],
#           [2, 3],
#           [3, 1],
#           [4, 3],
#           [5, 3],
#           [6, 2]]
# y_data = [[0],
#           [0],
#           [0],
#           [1],
#           [1],
#           [1]]


####### batch reading으로 추가된 코드들
filename_queue = tf.train.string_input_producer(['../DeepLearningZeroToAll-master/data-03-diabetes.csv'],
                                                shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_default = [[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_default) ## value는 Reader에서 읽어들인 값
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
#####################################3333


X = tf.placeholder(tf.float32, shape=[None, 8], name='TrainingData')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='TargetValue')
W = tf.Variable(tf.random_normal(shape=[8, 1]), name='Weight')
b = tf.Variable(tf.random_normal(shape=[1]), name='Bias')

H = tf.sigmoid(tf.matmul(X,W)+b)
cost = -tf.reduce_mean(Y*tf.log(H)+(1-Y)*tf.log(1-H))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

predict = tf.cast((H>0.5), dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

## 여기 추가됨
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
################

for epoch in range(10000):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch]) ## 만들어 놓은 그래프에서 데이터를 읽어온다.
    # loss_val, hypothesis_val, _ = sess.run([loss, hypothesis, train], feed_dict={X: x_data, Y: y_data}
    accu_val, cost_val,_ = sess.run([accuracy, cost, train], feed_dict={X:x_batch, Y:y_batch})
    if epoch%20==0:
        print("Accuracy=%s Cost=%s" %(accu_val,cost_val))

### 여기도 추가됨 (관례적으로 이렇게 한다고 외워둘 부분임)
coord.request_stop()
coord.join(threads)