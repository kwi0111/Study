

# x y 추출해서 모델 만드시오
# 성능 0.99이상

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten ,Input, Dropout, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator( # 이미지를 0~255의 값으로 정규화된 배열로 변환
    rescale=1./255,          # 시작부터 스케일링 하겠다. 0~1 사이
    horizontal_flip=True,    # 수평으로 뒤집겠다.
    vertical_flip=True,      # 수직으로 뒤집겠다.
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
# 트레인 데이터는 훈련을 해야하지만 테스트 데이터는 그대로 검증해야함. // 스케일링은 해줘야함 // 

path_train = 'c:/_data/image/brain/train/'
path_test = 'c:/_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(       # 지정된 디렉토리에서 이미지를 로드하고, 이를 모델에 주입할 수 있는 데이터 생성기를 생성
    path_train,
    target_size=(200, 200),     #사진 작으면 (200, 200)으로 커지고 크면 (200, 200)으로 작아진다 // 모든사진 (200, 200)
    batch_size=160,              # 160이상을 쓰면 x 통데이터로 가져올수 있다.
    class_mode='binary',
    shuffle=True,
)   # Found 160 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000020E66130670> 반복자 // x와 y가 합쳐져있는 형태
# DirectoryIterator 여기서 x는 (배치 크기, *표적 크기, 채널)의 형태의 이미지 배치로 구성된 numpy 배열이고 y는 그에 대응하는 라벨로 이루어진 numpy 배열
# 2 classes : ad, nomal

xy_test = test_datagen.flow_from_directory(       # 수치화
    path_test,
    target_size=(200, 200), 
    batch_size=120,
    class_mode='binary',
    # shuffle=True,
) 

print(xy_test)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000020E66130670> 반복자 // x와 y가 합쳐져있는 형태
print(xy_train.next())
# print(xy_train[0])  # array([0., 0., 0., 0., 1., 1., 1., 1., 1., 0.] = y값
# print(xy_train[16])  # 에러 :: 전체 데이터 / batch_size = 160/10 = 16개인데
                     # [16]은 17번째 값을 빼라고 하니 에러가 난다.

# print(xy_train[0][0])   # 첫번째 배치의 x
# print(xy_train[0][1])   # 첫번째 배치의 y
# print(xy_train[0][0].shape)   # (160, 200, 200, 3)



# print(type(xy_train))   # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))    # <class 'tuple'>
# print(type(xy_train[0][0]))    # <class 'numpy.ndarray'>
# print(type(xy_train[0][1].shape))    # <class 'numpy.ndarray'>

# print(xy_train[0][0].shape)   # (160, 200, 200, 3)
# print(xy_test[0][0].shape)   # (120, 200, 200, 3)

# print(xy_train[0][0])   # x_train (160, 200, 200, 3)
# print(xy_train[0][1])   # y_train (160,)
# print(xy_test[0][0])    # x_test (120, 200, 200, 3)
# print(xy_test[0][1])    # y_test (120,)

x_train = xy_train[0][0]    # (160, 200, 200, 3)
y_train = xy_train[0][1]    # (160,)
x_test = xy_test[0][0]      # (120, 200, 200, 3)
y_test = xy_test[0][1]      # (120,)
# print(pd.value_counts(y_test))

# x_train = x_train.reshape(160, 200*200*3)  
# x_test = x_test.reshape(120, 200*200*3)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(-1, 200, 200, 3)
# x_test = x_test.reshape(-1, 200, 200, 3)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


# ohe = OneHotEncoder(sparse = False)
# y_train = ohe.fit_transform(y_train.reshape(-1, 1))
# y_test = ohe.fit_transform(y_test.reshape(-1, 1))

print(y_train.shape)
print(y_test.shape)



#2. 모델
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', strides=2, input_shape = (200,200,3), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D())
# model.add(Conv2D(32, (2,2), padding='same', activation='relu' ))
model.add(Conv2D(32,(2,2), padding='same',  strides=2, activation='relu' ))
model.add(Dropout(0.1))
model.add(Conv2D(16, (2,2), padding='same', strides=2,  activation='relu' ))
model.add(Flatten())
model.add(Dense(4, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=500,
                verbose=1,
                restore_best_weights=True
                )
print(x_train.shape, y_train.shape)
model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=1, validation_split=0.2)

#4 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss', results[0])
print('acc', results[1])

# loss 0.6652514338493347
# acc 0.7749999761581421

# loss 0.4926069974899292
# acc 0.875

# loss 0.3672873377799988
# acc 0.9083333611488342

# loss 0.15552547574043274
# acc 0.9583333134651184




