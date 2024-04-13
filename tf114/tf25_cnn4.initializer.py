import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1. 데이터

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, [None,28,28,1])    # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None,10]) 

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[2,2,1,1],  # 커널사이즈 (2,2) / 컬러(채널 1개, 흑백) / 1개 필터(아웃풋) 
                                 initializer=tf.contrib.layers.xavier_initializer())
with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())
    w1_val = sess.run(w1)
    print(w1_val, '\n', w1_val.shape)



