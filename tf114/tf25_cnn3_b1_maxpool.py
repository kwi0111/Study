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
w1 = tf.compat.v1.get_variable('w1', shape=[2,2,1,128])  # 커널사이즈 (2,2) / 컬러(채널 1개) / 128개 필터(아웃풋) /
b1 = tf.compat.v1.Variable(tf.zeros([128]), name='b1')   # 필터의 갯수와 동일하다.

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID')    # 필터가 이미지를 한 픽셀씩 이동하며 적용
L1 = L1 + b1    # L1 += b1
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# model.add(Conv2d(64, kenel_size(2,2), stride=(1,1), input_shape=(28,28,1)))
print(L1)   # Tensor("Relu:0", shape=(?, 27, 27, 128), dtype=float32)
print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 13, 13, 128), dtype=float32)

'''
print(w1)   # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)   # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3,3,64,32]) # 커널, 채널, 필터
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')    # 필터가 이미지를 한 픽셀씩 이동하며 적용

print(L2)
'''

