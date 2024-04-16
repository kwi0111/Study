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

print(x_train[0])
print(y_train[0])
print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#   dtype=int64))
print(pd.value_counts(y_test))
x_train = x_train.reshape(60000, 28, 28, 1) # 데이터 내용 순서 변화 없다
# x_test = x_test.reshape(10000, 28, 28, 1) # 데이터 내용 순서 변화 없다
# print(x_train.shape[0]) # 60000

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # 데이터 내용 순서 변화 없다
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))


#2. 모델
model = Sequential()
model.add(Conv2D(100, (2,2), 
                 strides=1,
                 input_shape=(10, 10, 1),
                 padding='same',))                 
model.add(MaxPooling2D()) 
model.add(Conv2D(100, (2,2), padding='same') )
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=50)) # 주로 2차원 받음 
model.add(Dense(10, activation='softmax'))


print(model.summary())

# GAP
# Total params: 46,160 (180.31 KB)
#  Trainable params: 46,160 (180.31 KB)
#  Non-trainable params: 0 (0.00 B)

# Flatten
#  Total params: 166,160 (649.06 KB)
#  Trainable params: 166,160 (649.06 KB)
#  Non-trainable params: 0 (0.00 B)

'''

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

#4 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss', results[0])
print('acc', results[1])
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict, axis=1)
arg_test = np.argmax(y_test, axis=1)

print(arg_pre)

'''






