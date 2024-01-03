import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10)])   # 파이썬 = 언어, 넘파이 = API / 텐서플로우 친구,,   [[0 1 2 3 4 5 6 7 8 9]] 0부터 10-1 까지
print(x)
print(x.shape)  # (1, 10)

x = np.array([range(1, 10)])   #1부터 10-1 까지
print(x)    #[[1 2 3 4 5 6 7 8 9]]
print(x.shape)  # (1, 9)

x = np.array([range(10), range(21, 31), range(201, 211)])   
print(x)
print(x.shape)
x = x.transpose()
print(x)
print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])     
y = y.T
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim = 3))
model.add(Dense(140))
model.add(Dense(200))
model.add(Dense(160))
model.add(Dense(15))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print("로스 : ", loss)
print("[10, 31, 211]의 예측값 : ", results)



# 로스 :  6.51054765654635e-10
# [10, 31, 211]의 예측값 :  [[10.999944   2.0000398 -0.9999872]]








