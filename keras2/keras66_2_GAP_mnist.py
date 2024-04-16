# Flatten과 비교해서 설명해주세요. -> Flatten 대신쓴다. -> Flatten 문제점 뭐냐
# 어차피 계산해서 큰거 뽑을바에 그냥 큰거 바로 뽑자.
# 필터마다 에버리지풀링. -> 각각의 이미지 생성 -> Dense layer

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import OneHotEncoder


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape    )# (10000, 28, 28) (10000,)
print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#   dtype=int64))

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

#2. 모델
model = Sequential()
model.add(Conv2D(100, (2,2), 
                 strides=1,
                 input_shape=(28, 28, 1),
                 padding='same',))                 
model.add(MaxPooling2D())
model.add(Conv2D(100, (2,2), padding='same') )
model.add(Conv2D(100, (2,2), padding='same') )
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=50))
model.add(Dense(units=50))
model.add(Dense(10, activation='softmax'))
print(model.summary())



#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1, validation_split=0.2)

#4 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss', results[0])
print('acc', results[1])
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
# print(y_test)
print('acc_score : ', accuracy_score(y_test, y_predict))
# acc_score :  0.7095
