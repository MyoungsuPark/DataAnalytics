# https://www.youtube.com/watch?v=o2q4QNnoShY&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=10
# TensorFlow 파일에서 데이타 읽어오기 (new)
import tensorflow as tf
import numpy as np
# # data-01=
# 73	80	75	152
# 93	88	93	185
# 89	91	90	180
# 96	98	100	196
# 73	66	70	142
# 53	46	55	101
# 69	74	77	149
# 47	56	60	115
# 87	79	90	175
# 79	70	88	164
# 69	70	73	141
# 70	65	74	141
# 93	95	91	184
# 79	80	73	152
# 70	73	78	148
# 93	89	96	192
# 78	75	68	147
# 81	90	93	183
# 88	92	86	177
# 78	83	77	159
# 82	86	90	177
# 86	82	89	175
# 78	83	85	175
# 76	83	71	149
# 96	93	95	192

# 위 에디터를 읽고 결과를 예측하는 프로그램을 작성하라.
xy = np.loadtxt('../DeepLearningZeroToAll-master/data-01-test-score.csv', dtype=np.float32, delimiter=',')

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
y_data_ = xy[:, -1] #이렇게 하면 골아픈 에러 나온다.!!!!1

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal(shape=[3, 1]))
b = tf.Variable(tf.random_normal(shape=[1]))

hypothesis = tf.matmul(X, W) + b
print(hypothesis) # 값이 전혀 나오지 않음... Tensor라고만 나온다.
loss = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss=loss)

#실행단계
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(2001):
    loss_val, hypothesis_val,_ = sess.run([loss, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if epoch % 20 == 0:
        print("Epoch:%s \t LOSS=%s \t" % (epoch, loss_val))
        print("Hypothesis\n %s" % hypothesis_val)

## Prediction에 사용하기.
print(sess.run(hypothesis, feed_dict={X: [[10, 20, 30]]}))
print(sess.run(hypothesis, feed_dict={X: [[101, 90, 80], [90,30,50]]}))


################################################
# 파일이 너무 커서 한번에 메모리에 올릴 수 없는  경우
# 파일 배치 방법
tf.reset_default_graph()