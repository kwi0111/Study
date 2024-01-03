import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(400))
model.add(Dense(1000))
model.add(Dense(300))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([11000, 7])     
print("로스 : ", loss)
print("[11000, 7]의 예측값", results)

# 로스 :  0.20027343928813934
# [11000, 7]의 예측값 [[1.0804407e+05]
#  [6.9383831e+00]]

