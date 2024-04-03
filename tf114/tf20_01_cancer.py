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

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1223, train_size=0.8, stratify=y)
x_train = np.float32(x_train)
x_test = np.float32(x_test)
y_train = np.float32(y_train)
y_test = np.float32(y_test)


#2. 모델
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,50]), name='weight1', dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([50]), name='bias1', dtype=tf.float32)

layer1 = tf.compat.v1.matmul(xp, w1) + b1 

# layer2 : model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([50, 30], name='weight2'))
b2 = tf.compat.v1.Variable(tf.zeros([30], name='bias2'))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2    

# layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([30,10], name='weight3'))
b3 = tf.compat.v1.Variable(tf.zeros([10], name='bias3'))
layer3 = tf.compat.v1.matmul(layer2, w3) + b3    

# layer4 : model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([10,5], name='weight4'))
b4 = tf.compat.v1.Variable(tf.zeros([5], name='bias4'))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)  

# output_layer : model.add(Dense(1), activation = 'sigmoid')
w5 = tf.compat.v1.Variable(tf.random_normal([5,1], name='weight5'))
b5 = tf.compat.v1.Variable(tf.zeros([1], name='bias5'))
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)  



#3-1. 컴파일
loss = -tf.reduce_mean(yp * tf.log(hypothesis) + (1 - yp) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, yp),dtype=tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, _ = sess.run([loss, train], feed_dict = {xp:x_train, yp:y_train})
        
        if step % 200 == 0 :
            print(step, cost_val)
            
    hypo, pred, acc = sess.run([hypothesis, predicted, accuracy],
                               feed_dict = {xp:x_test, yp:y_test})
    print("훈련값 : ", hypo)
    print("예측값 : ", pred)
    print("정확도 : ", acc)

# 정확도 :  0.6315789








