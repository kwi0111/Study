from sklearn.datasets import load_diabetes
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
tf.set_random_seed(777)

#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target
y = y.reshape(-1,1) # (442,1) # 웨이트와 행렬연산을 해야하기 때문

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1223, train_size=0.8)


#2. 모델
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,32]), name='weight1', dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]), name='bias1', dtype=tf.float32)

layer1 = tf.compat.v1.matmul(xp, w1) + b1 

# layer2 : model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([32, 16], name='weight2'))
b2 = tf.compat.v1.Variable(tf.zeros([16], name='bias2'))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2    

# layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([16,1], name='weight3'))
b3 = tf.compat.v1.Variable(tf.zeros([1], name='bias3'))
hypothesis = tf.compat.v1.matmul(layer2, w3) + b3    

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp))
train = tf.train.GradientDescentOptimizer(learning_rate= 1e-5 ).minimize(loss)


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5000
for step in range(epochs):
    cost_val, _ = sess.run([loss, train],
                                         feed_dict = {xp:x_train, yp:y_train})
    if step % 200 == 0:
        print(step, 'loss : ', cost_val)

y_pred = sess.run(hypothesis, feed_dict={xp: x_test})
print('예측값:', y_pred)

mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"MSE = {mse:.4f}")
print(f"R2 score = {r2:.4f}")

sess.close()














