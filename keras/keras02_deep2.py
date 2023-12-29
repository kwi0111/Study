import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델 구성
#### [실습] 100epochs에 01_1번과 같은 결과를 빼시오.
#### 소수점 4째자리까지 맞춘다.
#### 로스값 0.33이하
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
model.compile(loss="mse", optimizer='adam')
model.fit(x, y, epochs =100)  # 에포 튜닝을 해야한다.

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([1,2,3,4,5,6,7])
print("7의 예측값 : ", result)


# 로스 :  0.32581883668899536
# 1/1 [==============================] - 0s 85ms/step
# 7의 예측값 :  [[1.0642405]
#  [2.0306938]
#  [2.9971468]
#  [3.9636002]
#  [4.930054 ]
#  [5.896508 ]
#  [6.862959 ]]