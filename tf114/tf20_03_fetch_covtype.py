from sklearn.datasets import fetch_covtype
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터 
datasets = fetch_covtype()
x_data = datasets.data
y_data = datasets.target
y_data = pd.get_dummies(y_data).values

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1223, train_size=0.8, stratify=y_data)
x_train = np.float32(x_train)
x_test = np.float32(x_test)
y_train = np.float32(y_train)
y_test = np.float32(y_test)


#2. 모델
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([54,128]), name='weight1', dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([128]), name='bias1', dtype=tf.float32)

layer1 = tf.compat.v1.matmul(xp, w1) + b1 

# layer2 : model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([128, 64], name='weight2'))
b2 = tf.compat.v1.Variable(tf.zeros([64], name='bias2'))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2    

# layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([64,32], name='weight3'))
b3 = tf.compat.v1.Variable(tf.zeros([32], name='bias3'))
layer3 = tf.compat.v1.matmul(layer2, w3) + b3    

# layer4 : model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([32,16], name='weight4'))
b4 = tf.compat.v1.Variable(tf.zeros([16], name='bias4'))
layer4 = tf.nn.softmax(tf.compat.v1.matmul(layer3, w4) + b4)

# output_layer : model.add(Dense(1), activation = 'sigmoid')
w5 = tf.compat.v1.Variable(tf.random_normal([16,7], name='weight5'))
b5 = tf.compat.v1.Variable(tf.zeros([7], name='bias5'))
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5)  



#3-1. 컴파일
loss = -tf.reduce_mean(yp * tf.log(hypothesis) + (1 - yp) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, yp),dtype=tf.float32))

epochs = 100
for step in range(epochs):
    cost_val, _ = sess.run([loss, train],
                                         feed_dict = {xp:x_train, yp:y_train})
    if step % 1 == 0:
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
# ACC :  0.95
sess.close()




















