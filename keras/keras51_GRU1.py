import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, SimpleRNN, LSTM, GRU


# 1.데이터
datesets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6,],
             [5,6,7],
             [6,7,8],
             [7,8,9],]
             )


y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) # (7, 3) (7,)
x = x.reshape(7,3,1)
print(x.shape, y.shape) # (7, 3, 1) (7,)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=2000, input_shape = (3, 1)))  #  units(아웃풋 갯수), (timesteps, features)
# Input shape : 3-D tensor with shape (batch_size, timesteps, features). (데이터갯수, 시간의 크기, 열의 갯수)
model.add(GRU(10, input_shape=(3,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))
model.summary()


""" #3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)

#4. 평가, 예측
results = model.evaluate(x, y)
print('로스 : ', results)
y_pred = np.array([8,9,10]).reshape(1,3,1)
y_pred = model.predict(y_pred)
# (3,) -> (1,3,1)로 바꾼다

print('[8,9,10,]의  결과 : ', y_pred) """










