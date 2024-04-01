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
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(loss)

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5).minimize(loss)

# [실습]
#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())   # 변수 초기화

epochs = 2000
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict = {x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss : ', cost_val)

print(w_val, b_val)

#4. 평가, 예측

predict = tf.argmax(tf.nn.softmax(tf.matmul(x, w_val) + b_val), axis=1)
actual = tf.argmax(y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, actual), dtype=tf.float32))

acc_val = sess.run(accuracy, feed_dict={x: x_data, y: y_data})
print('정확도:', acc_val)


# x_test = tf.compat.v1.placeholder(tf.float32, shape = [None, 4])

# y_pred = tf.sparse_softmax(tf.matmul(x_test, w_val) + b_val)
# y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={x_test:x_data})

# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# r2 = r2_score(y_data, y_predict)
# print('R2 SCORE : ' , r2) 
