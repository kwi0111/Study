import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터

x = np.array([range(10)])   
x = x.transpose()

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])
y = y.T

#.2 모델 구성
model = Sequential()
model.add(Dense(20, input_dim = 1))
model.add(Dense(40))
model.add(Dense(340))
model.add(Dense(140))
model.add(Dense(40))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=2)      

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10]])
print("로스 : ", loss)
print("[10]의 예측값 : ", results)


# 로스 :  9.73517598999718e-14
# [10]의 예측값 :  [[10.999997   2.0000002 -1.0000002]]


