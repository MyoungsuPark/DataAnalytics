# https://www.youtube.com/watch?v=o2q4QNnoShY&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=10
# 배치 파일에서 데이타 읽어오기

import tensorflow as tf

### 여러 파일에서 배치로 읽어들일때 아래 3줄 코드를 그대로 쓴다.(믿고 써라)
filename_queue = tf.train.string_input_producer(['../DeepLearningZeroToAll-master/data-01-test-score.csv'],
                                                shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
########

record_default = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_default) ## 디버깅해보면 xy는 Tensor list이다.총 4개의 Tensor list로 구성되어 있다.
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10) # 4개 텐서중. 1,2,3번째는 X   나머지 4번째는 Y
# train_x_batch, train_y_batch는 tf.train.batch가 실행되어야 비로소 값을 갖게된다. (Session 실행 부분에서 값을 채워줌)

### 아래 코드는 ML lab 04-2.py와 동일하다.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal(shape=[3, 1]))
b = tf.Variable(tf.random_normal(shape=[1]))

hypothesis = tf.matmul(X, W) + b
print(hypothesis) # 값이 전혀 나오지 않음... Tensor라고만 나온다.
loss = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss=loss)
###########################################

#########################################
## 실행 단계에서 조금 추가되는 부분이 있음.
#실행단계
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## 여기 추가됨
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
################

for epoch in range(2001):
    ##### 여기도 수정/추가됨
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch]) ## 만들어 놓은 그래프에서 데이터를 읽어온다.
    # loss_val, hypothesis_val, _ = sess.run([loss, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    loss_val, hypothesis_val, _ = sess.run([loss, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    #########################

    if epoch % 20 == 0:
        print("Epoch:%s \t LOSS=%s \t" % (epoch, loss_val))
        print("Hypothesis\n %s" % hypothesis_val)

### 여기도 추가됨 (관례적으로 이렇게 한다고 외워둘 부분임)
coord.request_stop()
coord.join(threads)