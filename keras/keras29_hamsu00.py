# 04_mlp에서 가져옴

import numpy as np
from keras.models import Sequential, Model # 함수형 모델
from keras.layers import Dense, Input, Dropout

# 백터 -> 행렬

#1. 데이터 
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]    
             )

y = np.array([1,2,3,4,5,6,7,8,9,10])

x = x.transpose()

#2. 모델 구성 (순차적)
# model = Sequential()
# model.add(Dense(10, input_shape=(2,)))
# model.add(Dense(9))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation = 'relu'))
# model.add(Dense(7))
# model.add(Dense(1))

#2. 모델 구성 (함수형) // 동일한 모델이다. 재사용하기 위해서 씀
input1 = Input(shape=(2,))
dense1 = Dense(10)(input1)  # 그 레이어(input1)의 인풋 레이어 
dense2 = Dense(9)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation = 'relu')(dense2)
drop1 = Dropout(0.2)(dense3)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs = output1)    # 모델 정의 -> 어디서부터 어디까지 

model.summary()     # 인풋 레이어부터 보여준다. 

# #3. 컴파일, 훈련
# model.compile(loss="mse", optimizer='adam')
# model.fit(x,y, epochs=100, batch_size=2)

# #4. 평가, 예측
# loss = model.evaluate(x, y)
# results = model.predict([[10, 1.3]]) 
# print("로스 : ", loss)
# print("[10, 1.3]의 예측값", results)
