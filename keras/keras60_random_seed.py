import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import random as rn


print(tf.__version__)   # 2.9.0
print(keras.__version__) # 2.9.0
print(np.__version__)   # 1.26.3
rn.seed(333)           # 파이썬 랜덤 씨드 고정
tf.random.set_seed(123) # 텐서 2.9.에서는 가능, 2.15안됌
np.random.seed(321)    # 보정
# 가중치 초기값 고정
# 원래는 가중치값 고정 x ->



#1.데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2.모델
model = Sequential()
model.add(Dense(5, 
                # kernel_initializer='zeros', # 가중치 0으로 초기화
                input_dim = 1))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100)

#4. 평가, 예측
loss = model.evaluate(x,y, verbose=1)
print('loss : ', loss)
results = model.predict([4], verbose=0)
print('4의 예측값 : ', results)

# tf2.9버젼 gpu 고정






