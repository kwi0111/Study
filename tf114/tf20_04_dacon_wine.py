from sklearn.datasets import fetch_covtype
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

#1.데이터 
path = "c:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv.shape)  # (5497, 13)

test_csv = pd.read_csv(path + 'test.csv', index_col = 0)    
print(test_csv.shape)  # (1000, 12)

submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(submission_csv.shape)  # (1000, 2)

train_csv['type'] = train_csv['type'].map({"white":1, "red":0}).astype(int)
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0}).astype(int)

# x와 y를 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
print(x.shape, y.shape)     # (5497, 12) (5497,)
print(np.unique(y, return_counts=True)) # (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
print(pd.value_counts(y))

# 원핫. 판다스
y_ohe = pd.get_dummies(y, dtype='int')
print(y_ohe.shape)  # (5497, 7)
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, random_state=1223, train_size=0.8, stratify=y_ohe)

scaler = MinMaxScaler() # 클래스 정의
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = np.float32(x_train)
x_test = np.float32(x_test)
y_train = np.float32(y_train)
y_test = np.float32(y_test)


#2. 모델
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 12])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([12,128]), name='weight1', dtype=tf.float32)
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


























