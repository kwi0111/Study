from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터 
datasets = load_iris()
x, y = datasets.data , datasets.target
print(x.shape, y.shape) # (150, 4) (150,)
x_data = x[y != 2]
y_data = y[y != 2]   # 2인 놈들 제외하겠다.
print(x_data.shape, y_data.shape)   # (100, 4) (100,)

y_data = y_data.reshape(-1, 1)
print(y_data.shape) # (100, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=123, train_size=0.8, stratify=y_data)

x_train = np.float32(x_train)
x_test = np.float32(x_test)
y_train = np.float32(y_train)
y_test = np.float32(y_test)

#2. 모델
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])


w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,1]), name='weight', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)


hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp, w) + b)

#3-1. 컴파일
loss = -tf.reduce_mean(yp * tf.log(hypothesis) + (1 - yp) * tf.log(1 - hypothesis))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2000
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict = {xp:x_train, yp:y_train})
    if step % 20 ==0:
        print(step, 'loss : ', cost_val)
    
# y_pred = sess.run(hypothesis, feed_dict={xp: x_test})
# print('예측값:', y_pred)

# #4. 평가, 예측
y_pred = tf.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={xp:x_test})
print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('ACC : ' , acc) 


sess.close()


