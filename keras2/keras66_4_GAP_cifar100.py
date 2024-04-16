# Flatten과 비교해서 설명해주세요. -> Flatten 대신쓴다. -> Flatten 문제점 뭐냐
# 어차피 계산해서 큰거 뽑을바에 그냥 큰거 바로 뽑자.
# 필터마다 에버리지풀링. -> 각각의 이미지 생성 -> Dense layer

import numpy as np
import pandas as pd
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import OneHotEncoder


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape    )   # (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True))  

#2. 모델
model = Sequential()
model.add(Conv2D(100, (2,2), 
                 strides=1,
                 input_shape=(32, 32, 3),
                 padding='same',))                 
model.add(MaxPooling2D())
model.add(Conv2D(100, (2,2), padding='same') )
model.add(Conv2D(100, (2,2), padding='same') )
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=50))
model.add(Dense(units=50))
model.add(Dense(units=50))
model.add(Dense(units=50))
model.add(Dense(100, activation='softmax'))
print(model.summary())


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=256, verbose=1, validation_split=0.2)

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

# GAP
# loss 3.5908820629119873
# acc 0.16120000183582306
# 313/313 [==============================] - 0s 722us/step
# acc_score :  0.1612

# FLA
# loss 4.271364212036133
# acc 0.20399999618530273
# 313/313 [==============================] - 0s 814us/step
# acc_score :  0.204