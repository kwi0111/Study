import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7]) 
y_train = np.array([1,2,3,4,6,5,7])       # 훈련에 쓴다

x_test = np.array([8,9,10]) # 이것으로 평가 하겠다. / 훈련 데이터와 평가 데이터를 나눈다. / 훈련값과 평가값이 비슷하면 신뢰 O
y_test = np.array([8,9,10]) # 로스값 훈련값이 평가보다 좋다 // 60% ~ 90%를 훈련 -> 나머지 테스트 // 나누는 이유 과적합 방지 // 


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

# 7/7 [==============================] - 0s 3ms/step - loss: 0.3087
# 1/1 [==============================] - 0s 82ms/step - loss: 0.0260
# 1/1 [==============================] - 0s 60ms/step
# 로스 :  0.026040107011795044      위의 로스보다 통상적으로 더 로스가 낮다.
# [11000, 7]의 예측값 [[1.0596814e+04]
#  [6.9147692e+00]]

# 