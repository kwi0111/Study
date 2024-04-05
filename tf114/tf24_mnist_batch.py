import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
tf.set_random_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델
rate = tf.compat.v1.placeholder('float32')
x = tf.compat.v1.placeholder('float32', shape=[None, 784])
y = tf.compat.v1.placeholder('float32', shape=[None, 10])

w1 = tf.compat.v1.get_variable('w1',shape=[784, 64],
                           initializer = tf.contrib.layers.xavier_initializer())

b1 = tf.compat.v1.Variable(tf.zeros([64]), name='b1')
layer1 = tf.compat.v1.matmul(x, w1) + b1 
layer2 = tf.compat.v1.nn.relu(layer1) 
layer1 = tf.compat.v1.nn.dropout(layer1, rate=rate)

# layer2 : model.add(Dense(9))
w2 = tf.compat.v1.get_variable('w2',shape=[64, 32],
                               initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.compat.v1.Variable(tf.zeros([32], name='b2'))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2 
layer2 = tf.compat.v1.nn.relu(layer2)   
layer2 = tf.compat.v1.nn.dropout(layer2, rate=rate)  
   

# layer3 : model.add(Dense(8))
w3 = tf.compat.v1.get_variable('w3',shape=[32, 16])
b3 = tf.compat.v1.Variable(tf.zeros([16], name='b3'))
layer3 = tf.compat.v1.matmul(layer2, w3) + b3
layer3 = tf.compat.v1.nn.relu(layer3)   
layer3 = tf.compat.v1.nn.dropout(layer3, rate=rate)  

# layer3 : model.add(Dense(8))
w4 = tf.compat.v1.get_variable('w4',shape=[16, 10])
b4 = tf.compat.v1.Variable(tf.zeros([10], name='b4'))
layer4 = tf.compat.v1.matmul(layer3, w4) + b4
hypothesis = tf.compat.v1.nn.softmax(layer4)

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.nn.log_softmax(hypothesis, axis=1)))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

training_epochs = 101
batch_size = 100
total_batch = int(len(x_train) / batch_size)    # 600
# 60000 / 100
print(total_batch)


for step in range(training_epochs):
    
    avg_cost =0
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y, rate:0.1}
    
        cost_val, _, w_val, b_val = sess.run([loss, train, w4, b4], feed_dict = feed_dict) 
        avg_cost += cost_val / total_batch
        if step % 20 == 0 :
                print(step, 'loss :' , avg_cost)
       
   
#4. 평가, 예측
print("=======================")
y_predict = sess.run(hypothesis, feed_dict={x:x_test, rate:0.1})
y_predict_arg = sess.run(tf.arg_max(y_predict, 1))
y_test = np.argmax(y_test, 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict_arg, y_test)
print('ACC : ', acc)
# ACC :  0.9426
