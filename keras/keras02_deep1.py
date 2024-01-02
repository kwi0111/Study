from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))     #model.add : 레이어를 추가한다 / 노드(뉴런)의 개수, 레이어(층) 깊이 조절
model.add(Dense(15))
model.add(Dense(80))
model.add(Dense(30))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x,y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([4])
print("4의 예측값 : ", result)