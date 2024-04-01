import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  # (96294, 14)
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape)  # (64197, 13)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  # (64197, 2)

# 라벨 엔코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함

print(train_csv.describe)
print(test_csv.describe)

print(train_csv.shape)
print(test_csv.shape)
print(train_csv.dtypes)
print(test_csv.dtypes)

# x와 y를 분리
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']
print(x.shape, y.shape) # (96294, 13) (96294,)
print(np.unique(y, return_counts=True))

y = y.values.reshape(-1, 1)
print(y.shape)
# y = y.reshape(-1,1)   # 외않댐?
ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y)
print(y.shape)  

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y_ohe,             
                                                    train_size=0.86,
                                                    random_state=2024,
                                                    stratify=y,
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

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
w = tf.compat.v1.Variable(tf.random_normal([13, 7]), name = 'weight')
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
# ACC :  0.34913217623498
sess.close()
















