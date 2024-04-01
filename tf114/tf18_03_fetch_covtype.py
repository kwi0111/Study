from sklearn.datasets import fetch_covtype
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1.데이터 
datasets = fetch_covtype()

x_data = datasets.data
y_data = datasets.target
y_data = pd.get_dummies(y_data).values
print(x_data.shape, y_data.shape)   # (581012, 54) (581012, 7)


#2. 모델
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=123, train_size=0.8, stratify=y_data)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = np.float32(x_train)
x_test = np.float32(x_test)
y_train = np.float32(y_train)
y_test = np.float32(y_test)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
w = tf.compat.v1.Variable(tf.random_normal([54, 7]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1, 7]), name = 'bias')
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(xp, w) + b)

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(hypothesis), axis = 1)) 

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-1)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1000
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict = {xp:x_train, yp:y_train})
    if step % 20 == 0:
        print(step, 'loss : ', cost_val)


y_predict = sess.run(hypothesis, feed_dict = {xp:x_test})
print(y_predict)   
y_predict = sess.run(tf.argmax(y_predict, 1))
print(y_predict) 
y_data_arg = np.argmax(y_test, 1)
print(y_data_arg)    

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data_arg)
print('ACC : ', acc) 
# ACC :  0.5969897506949046
sess.close()






