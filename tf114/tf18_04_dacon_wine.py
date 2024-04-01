import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
print(y_ohe)
print(y_ohe.shape)  # (5497, 7)
print(test_csv)

x_train, x_test, y_train, y_test = train_test_split(
x, y_ohe,             
train_size=0.7,
random_state=123,
stratify=y_ohe,  
shuffle=True,
)
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = np.float32(x_train)
x_test = np.float32(x_test)
y_train = np.float32(y_train)
y_test = np.float32(y_test)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 12])
w = tf.compat.v1.Variable(tf.random_normal([12, 7]), name = 'weight')
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
# ACC :  0.5012121212121212
sess.close()











