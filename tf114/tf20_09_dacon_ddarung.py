# 7 diabetes
# 8 california
# 9 dacon 따릉이
# 10 kaggle bike

import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , r2_score
import pandas as pd
import numpy as np

#1. 데이터
x,y = fetch_california_housing(return_X_y=True)
path = 'C:\_data\dacon\ddarung\\'
train_csv = pd.read_csv(path + 'train.csv' , index_col=0 )
test_csv = pd.read_csv(path + 'test.csv' , index_col=0 )

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1)
y = train_csv['count']

x_train , x_test , y_train ,y_test = train_test_split(x,y, test_size=0.2 , shuffle=True , random_state=777 )

y_train = np.array(y_train)
y_train = y_train.reshape(-1,1)

xp = tf.compat.v1.placeholder(tf.float64 , shape= [None,9])
yp = tf.compat.v1.placeholder(tf.float64 , shape= [None,1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([9, 10], dtype=tf.float64))
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], dtype=tf.float64))
layer1 = tf.matmul(xp, w1) + b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 5], dtype=tf.float64))
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([5], dtype=tf.float64))
layer2 = tf.matmul(layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([5, 1], dtype=tf.float64))
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float64))
hypothesis = tf.matmul(layer2, w3) + b3

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp))

optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
for step in range(epochs) : 
    _, loss_val = sess.run([train , loss], feed_dict={xp : x_train , yp : y_train } )
    if step % 20==0 :
        print(step , '\t' ,loss_val )

#4 평가
# pred = tf.compat.v1.matmul(x_test,w_val) + b_val
predict = sess.run(hypothesis , feed_dict={xp:x_test})

r2 = r2_score(y_test, predict)
mae = mean_absolute_error(y_test, predict)
print('r2 : ',r2)
print('mae : ',mae)

sess.close()

# r2 :  -51.87307389012977
# mae :  510.77365405223696





