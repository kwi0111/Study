import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# [실습] 넘파이 리스트의 슬라이싱 7:3으로 잘라라.   [a:b:c] a 시작값, b 도착값, c 간격
x_train = x[:7]
y_train = y[0:7]        # 0부터 시작 // 7 도착

x_test = x[7:]
y_test = y[7:10]        # 7부터 시작 // 10 도착

print(x_train)  # [1 2 3 4 5 6 7]
print(y_train)  # [1 2 3 4 6 5 7]
print(x_test)   # [ 8  9 10]  
print(y_test)   # [ 8  9 10]


'''
#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(400))
model.add(Dense(1000))
model.add(Dense(300))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([11000, 7])     
print("로스 : ", loss)
print("[11000, 7]의 예측값", results)
'''