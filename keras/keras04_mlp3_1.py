import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10)])   # 파이썬 = 언어, 넘파이 = API (중간 전달자?) / 텐서플로우 친구,,   [[0 1 2 3 4 5 6 7 8 9]] 0부터 10-1 까지
print(x)                                    
print(x.shape)  # (1, 10)

x = np.array([range(1, 10)])   #1부터 10-1 까지
print(x)    #[[1 2 3 4 5 6 7 8 9]]
print(x.shape)  # (1, 9)

x = np.array([range(10), range(21, 31), range(201, 211)])   
print(x)
print(x.shape) # (3, 10)
x = x.transpose()
print(x)
print(x.shape) # (10, 3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],])     # 속성 3개로 y를 찾아내. / 대괄호[] 안에 있는것 : '두개 이상은' 리스트 / []안에 있는것들을 넘파이에 집어 넣음.
y = y.T
print(y.shape)  # (10, 2)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim = 3))
model.add(Dense(40))
model.add(Dense(800))
model.add(Dense(105))
model.add(Dense(2))     # y값 : 2개 리스트

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print("로스 : ", loss)
print("[10, 31, 211]의 예측값 : ", results)

# 배치 사이즈 : 3
# 로스 :  2.1934988581051584e-06
# [10, 31, 211]의 예측값 :  [[10.997597   1.9980247]]

# 배치 사이즈 : 4
# 로스 :  3.119629639058985e-07
# [10, 31, 211]의 예측값 :  [[10.999161   1.9989536]]

# 배치 사이즈 : 4 / Dense 80 → 800 / 15 → 105 
# 로스 :  2.4543921582631523e-12
# [10, 31, 211]의 예측값 :  [[10.999996   2.0000012]]



# 배운다 : 가중치를 긋고 로스를 줄이는것
# 예측 : [10, 31, 211]