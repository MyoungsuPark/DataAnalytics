# https://www.youtube.com/watch?v=fZUV3xjoZSM&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=9

import tensorflow as tf

# # H(X) = XW + b
#
# x1_data = [73, 93, 89, 96, 73]
# x2_data = [80, 88, 91, 98, 66]
# x3_data = [75, 93, 90, 100, 70]
# y_data = [152, 185, 180, 196, 142]
#
# # 데이터가 어떤게 들어 있는지 모른다 치고.
# # 이런 데이터를 런타임에 받아들이려면 tf.placeholder/feed_dict를 사용한다.
#
# x1 = tf.placeholder(tf.float32, name='x1')
# x2 = tf.placeholder(tf.float32, name='x2')
# x3 = tf.placeholder(tf.float32,name='x3')
# Y = tf.placeholder(tf.float32, name='Y')


#####
# matrix 모르면 위처럼 코딩했을 것이고. matrix 사용해 아래처럼 한다.
# shape [5,3]
x_data = [
    [73, 80, 75],
    [93, 88, 93],
    [89, 91, 90],
    [96, 98, 100],
    [73, 66, 70]
]
# shape [5,1]
y_data = [[152], [185], [180], [196], [142]]


# 그래프 만들기.
# 1. 데이터 인터페이스(placeholder), 텐서플로우로 계산해야할 변수(Variable)
X = tf.placeholder(tf.float32, shape=[None, 3], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
W = tf.Variable(initial_value=tf.random_normal([3, 1]), name="weight") # Variable (TF가 계산을 해주는 값) # W의 Shape는 [X_feature, Y_feature]
b = tf.Variable(initial_value=tf.random_normal([1]), name='bias') # Variable (TF가 계산을 해주는 값) #b의 shape는 [Y_feature]
# 가설 및 Cost 함수
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis-Y))
# 그래프 완료
###

###
# 옵티마이저 선언 (나중에 실행레벨에서 사용)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) # Learning Rate 0.001로 사용하면 결과 안나온다.
train = optimizer.minimize(loss=cost)  # 이것은 TF의 Operation 이다. (java 같은 대입 연산이 아님)

###
# 변수 초기화 및 실행레벨
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(2000):
    cost_val, hypothesis_val, train_val = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if epoch % 20 ==0:
        print(epoch, "\n cost ", cost_val, "\n Prediction \n", hypothesis_val)

