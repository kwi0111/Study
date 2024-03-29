from sklearn.datasets import load_iris, load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터 
datasets = load_breast_cancer()
x, y = datasets.data , datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

y = y.reshape(-1, 1)
print(y.shape) # (569, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, stratify=y)
x_train = np.float32(x_train)
x_test = np.float32(x_test)
y_train = np.float32(y_train)
y_test = np.float32(y_test)

#2. 모델
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])


w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1]), name='weight', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)


hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp, w) + b)

#3-1. 컴파일
loss = -tf.reduce_mean(yp * tf.log(hypothesis) + (1 - yp) * tf.log(1 - hypothesis))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20000
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict = {xp:x_train, yp:y_train})
    if step % 20 ==0:
        print(step, 'loss : ', cost_val)
    

y_pred = tf.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={xp:x_test})
print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('ACC : ' , acc) 


sess.close()


