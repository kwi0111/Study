# 62_2번의 결과를 손으로 계산하시오.
# 0 에포의 결과

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

#1. 데이터 
x = np.array([1,2])
y = np.array([1,2])

#2. 모델
model = Sequential()
model.add(Dense(2, input_dim = 1))
# model.add(Dense(2))
model.add(Dense(1))

###################################################
model.trainable = False # ★★★ 전이학습 할때 쓴다.
# model.trainable = True # 디폴트다
print("===========================================")
print(model.weights)
print("===========================================")
###################################################
# model.summary()

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x,y,batch_size=1, epochs=500, verbose=0)

#4. 평가, 예측
y_pred = model.predict(x)
print(y_pred)












