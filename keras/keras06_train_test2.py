import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# [실습] 넘파이 리스트의 슬라이싱 7:3으로 잘라라.   [a:b:c] a 시작값, b 도착값, c 간격 
x_train = x[:7]         # 0번부터 6번 // 명시하지 않으면 시작값 // [0:7] == [:-3]
y_train = y[0:7]        # 0부터 시작 // 7 도착
'''
a = b    # a라는 변수에 b값을 넣어라 
a == b   # a와 b가 같다.
'''
x_test = x[7:]          # [7:10] == [-3:] == [-3:10]
y_test = y[7:10]        # 7부터 시작 // 10 도착

print(x_train)  # [1 2 3 4 5 6 7]
print(y_train)  # [1 2 3 4 6 5 7]
print(x_test)   # [ 8  9 10]  
print(y_test)   # [ 8  9 10]

'''
평가데이터는 가중치에 영향 x // 백만개중에 뒤에 30만개 빼면 데이터 박살 // 임의로 30퍼 빼는게 낫다. 

'''




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