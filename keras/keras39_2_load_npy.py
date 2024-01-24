import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten ,Input, Dropout, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split

#1. 데이터 // 땡겨오는 시간만 좀 걸리지 연산속도는 빠름
np_path = 'c:/_data/_save_npy/'
# np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0])    # 넘파이 형태로 쏙 들어간다.
# np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])    # 넘파이 형태로 쏙 들어간다.
# np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0])    # 넘파이 형태로 쏙 들어간다.
# np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])    # 넘파이 형태로 쏙 들어간다.

x_train = np.load(np_path + 'keras39_1_x_train.npy')
y_train = np.load(np_path + 'keras39_1_y_train.npy')
x_test = np.load(np_path + 'keras39_1_x_test.npy')
y_test = np.load(np_path + 'keras39_1_y_test.npy')

print(x_train)
print(x_train.shape,y_train.shape)    # (160, 100, 100, 1) (160,)
print(x_test.shape, y_test.shape)    # (120, 100, 100, 1) (120,)

#2. 모델
model = Sequential()
model.add(Conv2D(88, (2,2), padding='same', strides=1, input_shape = (100,100,1), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), padding='same', activation='relu' ))
model.add(Conv2D(44, (2,2), padding='same',  strides=1, activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc',
                mode='min',
                patience=30,
                verbose=1,
                restore_best_weights=True
                )
# print(x_train.shape, y_train.shape)
model.fit(x_train, y_train, epochs=10,
                    # steps_per_epoch=16, # 전체 데이터 / batch = 160 / 10 = 16 /// 17이면 에러, 15면 나머지 소실
                    batch_size=32,    # fit_generator에서는 에러, fit에서는 안먹힘.
                    verbose=1,
                    validation_split=0.2, # 에러 
                    # validation_data=xy_test,
                    )

#4 평가, 예측
results = model.evaluate(x_test, y_test) 
print('loss', results[0])
print('acc', results[1])







