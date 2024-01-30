import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, SimpleRNN, LSTM, Dropout, Bidirectional


# 1.데이터
datesets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6,],
             [5,6,7],
             [6,7,8],
             [7,8,9],
             [8,9,10],
             [9,10,11],
             [10,11,12],
             [20,30,40],
             [30,40,50],
             [40,50,60]]
             )
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape) # (13, 3) (13,)
x = x.reshape(13,3,1)
print(x.shape, y.shape) # (13, 3, 1) (13,)

#2. 모델구성
model = Sequential()
model.add(Bidirectional(LSTM(100,), input_shape = (3, 1)))  #  units(아웃풋 갯수), (timesteps, features)
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)
x_predict = np.array([50,60,70]).reshape(1,3,1)
print(x_predict.shape)  # (1, 3, 1)
y_pred = model.predict(x_predict)
print('[50,60,70,]의  결과 : ', y_pred)

# loss :  0.006133092101663351
# (1, 3, 1)
# 1/1 [==============================] - 0s 208ms/step


# loss :  0.0013794555561617017
# [50,60,70,]의  결과 :  [[79.097145]]

