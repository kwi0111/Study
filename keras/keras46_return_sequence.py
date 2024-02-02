import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, SimpleRNN, LSTM, Dropout


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
x = x.reshape(-1,3,1)
print(x.shape, y.shape) # (13, 3, 1) (13,)

#2. 모델구성   
# return_sequences=True 모든 시점의 은닉 상태와 셀 상태를 리턴
# ex ) return_sequences=False,  hidden state (1, 3) -> (1, 3)  // return_sequences=True, hidden state (1, 4, 3) -> (1, 3)
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape = (3, 1), activation='sigmoid'))  #  2차원인데 밑에는 3차원을 받는다. 실행하려면 return_sequences=True 
model.add(LSTM(30, return_sequences=True, activation='sigmoid'))    # 두개 이상이라고해서 좋아진다는 보장 x
model.add(LSTM(100, return_sequences=True, activation='sigmoid'))
model.add(LSTM(10, activation='sigmoid'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))

model.summary()



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

# loss :  0.0004233964718878269
# (1, 3, 1)
# 1/1 [==============================] - 0s 361ms/step
# [50,60,70,]의  결과 :  [[77.26171]]

# loss :  0.0003340768744237721
# (1, 3, 1)
# 1/1 [==============================] - 1s 518ms/step
# [50,60,70,]의  결과 :  [[77.09387]]

# loss :  4.8414221964776516e-05
# (1, 3, 1)
# 1/1 [==============================] - 0s 372ms/step
# [50,60,70,]의  결과 :  [[76.56714]]

