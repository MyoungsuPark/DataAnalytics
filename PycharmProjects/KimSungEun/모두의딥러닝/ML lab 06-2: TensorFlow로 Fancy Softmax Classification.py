# https://www.youtube.com/watch?v=E-io76NlsqA&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=17


# 백그라운드
# logits = tf.matnul(X,W)+b
# hypo = tf.nn.softmax(logits)
# softmax는 확률을 의미한다.

# 이것을 간단히 하기위한 방법
# softmax_cross_entropy_with_logits
# 매개변수 logit 자리에 위의 logis값(tf.matmul(X,W)+b)를 넣어준다.

import numpy as np
import tensorflow as tf
xy = np.loadtxt('../DeepLearningZeroToAll-master/data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data = xy[:,:-1]
y_data = xy[:,[-1]] ## xy[:,-1] 이렇게 하면 에러 난다.
nb_classes = 7
x_features = x_data.shape[-1]

X = tf.placeholder(dtype=tf.float32, shape=[None,x_features])
Y = tf.placeholder(dtype=tf.int32, shape=[None,1])
Y_one_hot = tf.one_hot(Y,nb_classes)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes]) #one hot을 하면 차원이 하나 더 늘어나서 이것을 맞춰줘야 한다.

W = tf.Variable(initial_value=tf.random_normal(shape=[x_features,nb_classes]), dtype=tf.float32)  #W의 Shape는 [X_feature갯수, Y_Feature갯수]. ※Samples 영향은 받지 않는다.
b = tf.Variable(initial_value=tf.random_normal(shape=[nb_classes]), dtype=tf.float32) #

logits = tf.matmul(X,W)+b
hypo = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


predict = tf.arg_max(hypo, dimension=1)
Y_index = tf.arg_max(Y_one_hot, dimension=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y_index),dtype=tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(200000):
    predic_val, accu_val, cost_val, _ = sess.run([predict, accuracy, cost, train], feed_dict={X: x_data, Y:y_data})
    if epoch % 100==0:
        print("Epoch:%s. Cost:%s" %(epoch, cost_val))
        print("Accuracy= %s" %accu_val)


