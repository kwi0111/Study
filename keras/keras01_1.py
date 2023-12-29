import tensorflow as tf #tensorflow를 땡겨오고,  tf라고 쓴다.
print(tf.__version__) # 2.15.0
from tensorflow.keras.models import Sequential # tensorflow.keras.models에서 Sequential를 땡겨와주세여
from tensorflow.keras.layers import Dense      # 노란색줄 : 경고상태(실행은 됨)
import numpy as np

#1. 데이터
x = np.array([1,2,3])  #numpy 데이터 형식의 123 행렬
y = np.array([1,2,3])
# 테스트
#2. 모델 구성
model = Sequential()    #  순차적인 모델
model.add(Dense(20, input_dim=1))   
model.add(Dense(40))    
model.add(Dense(60))    
model.add(Dense(40))    
model.add(Dense(1))

#3. 컴파일, 훈련 (최적의 weigt)
model.compile(loss='mse', optimizer='adam')   # loss = 실제값과 예측값 차이. (절대값)낮을수록 좋다 / mse = 제곱하는 방법
model.fit(x, y, epochs=10000)     # fit 훈련 / epochs 훈련량 -> 최적의 웨이트가 생성

#4. 평가, 예측 (최소의 loss)
loss = model.evaluate(x, y)     # 로스값 평가
print("로스 : ", loss)
result = model.predict([4])
print("4의 예측값 : ", result)


# 로스 :  0.0
# 1/1 [==============================] - 0s 67ms/step
# 4의 예측값 :  [[3.9999986]]