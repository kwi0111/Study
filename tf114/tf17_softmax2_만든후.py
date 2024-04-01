import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

#1.데이터 
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],  # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],  # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],  # 0
          [1,0,0],
          ]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.compat.v1.Variable(tf.random_normal([4, 3]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1, 3]), name = 'bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

#2. 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)  # 회귀 모델, 단층 레이어

#3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) 
# Categorical_Cross_Entropy

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-1)
train = optimizer.minimize(loss)

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5).minimize(loss)

# [실습]
#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())   # 변수 초기화

epochs = 20000
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict = {x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss : ', cost_val)

print(w_val)
# [[-0.8206971  -0.18240486 -0.79235995]
#  [-0.13327104 -0.40440026  1.4131645 ]
#  [ 0.7955494   0.03625951  0.44701275]
#  [-0.2869433   0.5510848  -0.15126812]]
print(b_val)
# [[ 0.04705068  0.05709704 -0.10414778]]

#4. 평가, 예측

y_predict = sess.run(hypothesis, feed_dict = {x:x_data})
print(y_predict)    # 8행 3열 데이터
# y_predict = sess.run(tf.argmax(y_predict, 1))
# print(y_predict)    # [2 2 1 1 2 2 2 2]
y_predict = np.argmax(y_predict, 1)
print(y_predict)    #  [2 2 1 1 2 2 2 2]

y_data = np.argmax(y_data, 1)
print(y_data)    # [2 2 2 1 1 1 0 0]

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)
print('ACC : ', acc)    # ACC :  1.0

sess.close()


