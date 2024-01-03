# from tensorflow.keras.models. import Seqeuntial
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras
print("tf  버전 : ", tf.__version__) 
print("keras 버전 : ", keras.__version__)

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1840))
model.add(Dense(840))
model.add(Dense(640))
model.add(Dense(440))
model.add(Dense(140))
model.add(Dense(280))
model.add(Dense(60))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)       # 배치 단위로 잘라서 사용하겠다. 터질수가 있다. 일괄처리. 1에포 당 3번 돈다. 기본 32(2^5)
                                                # x : 학습 데이터, y : 레이블 데이터, 
                                                # fit : 훈련, epochs : 훈련량(전체 데이터 셋에 대한 반복 횟수), bitch_size : 몇 개의 샘플?
#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([7])
print("로스 : ", loss)
print("7의 예측값 : ", results)


# 배치 0
# 로스 :  0.324300080537796
# 7의 예측값 :  [[6.850133]]

# 배치 1
# 로스 :  1.1220927238464355
# 7의 예측값 :  [[5.0277185]]

# 배치 2
# 로스 :  0.3357609510421753
# 7의 예측값 :  [[6.59432]]

# 배치 3
# 로스 :  0.3378223478794098
# 7의 예측값 :  [[6.572144]]

# 배치4
# 로스 :  0.3248308002948761
# 7의 예측값 :  [[6.7548547]]

# 배치 5
# 로스 :  0.33249297738075256
# 7의 예측값 :  [[6.9989142]]