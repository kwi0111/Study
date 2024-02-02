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

#1. 데이터
train_datagen = ImageDataGenerator( # 이미지를 0~255의 값으로 정규화된 배열로 변환 // 이미지 숫자로 바꿔줌 // 컴퓨터가 알아먹을수있게 // 데이터 수치화, 변환하는 도구
    rescale=1./255,          # 시작부터 스케일링 하겠다. 0~1 사이
    # horizontal_flip=True,    # 수평으로 뒤집겠다.
    # vertical_flip=True,      # 수직으로 뒤집겠다.
    # width_shift_range=0.1,   # 0.1만큼 평행이동 하겠다.
    # height_shift_range=0.1,  # 0.1만큼 수직이동 하겠다.
    # rotation_range=5,        # 정해진 각도만큼 이미지를 회전
    # zoom_range=1.2,          # 축소 또는 확대
    # shear_range=0.7,         # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 ?.?
    fill_mode='nearest',     # 다른값 더 있음. // 옆에 최종값과 유사한값으로 잡아준다.
)

test_datagen = ImageDataGenerator(
    rescale=1./255, 
)
# predict에서 문제집 변경 x // 스케일링은 해야함 //

path_train = 'c:/_data/image/brain/train/'
path_test = 'c:/_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(  
# DirectoryIterator 여기서 x는 (배치 크기, *표적 크기, 채널)의 형태의 이미지 배치로 구성된 numpy 배열이고 y는 그에 대응하는 라벨로 이루어진 numpy 배열
    path_train,
    target_size=(100, 100),     
    batch_size=200,         # 몇장씩 수치화 할거냐 // 통배치
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)   # Found 160 images belonging to 2 classes.


xy_test = test_datagen.flow_from_directory(       # 수치화
    path_test,
    target_size=(100, 100), 
    batch_size=200,                                # 통배치
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
) 
# Found 160 images belonging to 2 classes.

print(xy_train[0][0].shape)   # x값 (160, 100, 100, 1) 
print(xy_train[0][1].shape)   # y값 (160,)

print(xy_test[0][0].shape)   # (120, 100, 100, 1)
print(xy_test[0][1].shape)   # (120,)


x = xy_train[0][0]    # 
y = xy_train[0][1]    # 

np_path = 'c:/_data/_save_npy/'
np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0])    # 넘파이 형태로 쏙 들어간다.
np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])    # 넘파이 형태로 쏙 들어간다.
np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0])    # 넘파이 형태로 쏙 들어간다.
np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])    # 넘파이 형태로 쏙 들어간다.


'''
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
model.fit(xy_train, epochs=10,
                    steps_per_epoch=16, # 전체 데이터 / batch = 160 / 10 = 16 /// 17이면 에러, 15면 나머지 소실
                    # batch_size=32,    # fit_generator에서는 에러, fit에서는 안먹힘.
                    verbose=1,
                    # validation_split=0.2, # 에러 
                    validation_data=xy_test,
                    )
#  UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. 
#  Please use `Model.fit`, which supports generators.
#4 평가, 예측
results = model.evaluate(xy_test)   # 사람이 평가
print('loss', results[0])
print('acc', results[1])
# print("걸린시간 : ", round(end - start, 2),"초")

'''
'''

텐서플로우 3번 문제
# loss 0.6632859706878662
# acc 0.8082143068313599





'''
