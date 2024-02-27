import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
# tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

#1. 데이터 
x = np.array([1,2,3,4,5,])
y = np.array([1,2,3,4,5,])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
# Non-trainable params: 0 은 뭐냐

print(model.weights)
#  numpy=array([[ 0.47288632, -0.78825045,  1.2209238 ]] 랜덤을 주어진 가중치의 초기값이요 // numpy=array([0., 0., 0.], 바이어스의 초기값 0이다.
# 'dense/kernel:0' 커널은 가중치다.
print(model.weights)
print("===========================================")
print(model.trainable_weights)
print("===========================================")
print(len(model.weights))   # 6
print(len(model.trainable_weights)) # 6 // 한레이어에 2개씩 있다 ( 가중치 하나 바이어스 하나)

###################################################
model.trainable = False # ★★★ 전이학습 할때 쓴다.
###################################################

print(len(model.weights))   # 6
print(len(model.trainable_weights)) # 0 // 훈련을 시키지 않겠다.

print("===========================================")
print(model.weights)
print("===========================================")
print(model.trainable_weights)

model.summary()
# Trainable params: 0
# Non-trainable params: 17 // 훈련 자체를 안시킨다.


