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
    horizontal_flip=True,    # 수평으로 뒤집겠다.
    vertical_flip=True,      # 수직으로 뒤집겠다.
    width_shift_range=0.1,   # 0.1만큼 평행이동 하겠다.
    height_shift_range=0.1,  # 0.1만큼 수직이동 하겠다.
    rotation_range=5,        # 정해진 각도만큼 이미지를 회전
    zoom_range=1.2,          # 축소 또는 확대
    shear_range=0.7,         # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 ?.?
    fill_mode='nearest',     # 다른값 더 있음. // 옆에 최종값과 유사한값으로 잡아준다.
)

# test_datagen = ImageDataGenerator(
#     rescale=1./255, 
# )
# predict에서 문제집 변경 x // 스케일링은 해야함 //

path_train = 'c:/_data/cat_and_dog//Train/'

xy_train = train_datagen.flow_from_directory(  
# DirectoryIterator 여기서 x는 (배치 크기, *표적 크기, 채널)의 형태의 이미지 배치로 구성된 numpy 배열이고 y는 그에 대응하는 라벨로 이루어진 numpy 배열
    path_train,
    target_size=(80, 80),     
    batch_size=5000,         # 몇장씩 수치화 할거냐     
    class_mode='binary',
    shuffle=True,
)   
x_train = xy_train[0][0]    # 
y_train = xy_train[0][1]    # 

# print(x_train.shape)
# print(y_train.shape)


augumet_size = 10000      # 가상 이미지의 수 (변수)

randidx = np.random.randint(x_train.shape[0], size=augumet_size)    # 50000중에서 15000개의 숫자를 뽑아내라.
                                                                    # np.random.randint(60000, 20000)
print(randidx)  # [  872 23666  7905 ... 17524 37111 22648]
print(np.min(randidx), np.max(randidx)) # 4 49999
x_augumented = x_train[randidx].copy()  # 메모리 원데이터에 영향을 미치지 않기위해서 사용 // 안전한 작업 수행
y_augumented = y_train[randidx].copy()   # 키벨류로 쌍이 맞음
print(x_augumented)
print(x_augumented.shape)   # (10000, 80, 80, 3)
print(y_augumented)         # [7 7 5 ... 2 7 1]
print(y_augumented.shape)   # (10000,)

# x_augumented = x_augumented.reshape(-1, 28, 28, 1)
# x_augumented = x_augumented.reshape(
#     x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 3)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augumet_size,
    shuffle=False,
    
)[0][0]

# x1, y1 = x_augumented[0] #x1 data의 0~999까지 // y1 0~999
# x2 = x_augumented[1][0] 

print(x_augumented)
print(x_augumented.shape)   # (10000, 80, 80, 3)


print(x_train.shape)    # (5000, 80, 80, 3)
# x_train = x_train.reshape(-1, 32, 32, 3)   
# x_test = x_test.reshape(-1, 32, 32, 1)

x_train = np.concatenate((x_train, x_augumented))        # 사슬 같이 잇다
y_train = np.concatenate((y_train, y_augumented))        # 사슬 같이 잇다

print(x_train.shape, y_train.shape) # (15000, 80, 80, 3) (15000,)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 1) (10000, 1)


ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
# y_test = ohe.transform(y_test.reshape(-1,1)).toarray()
print(y_train.shape)  # (15000, 2)
unique, counts = (np.unique(y_train, return_counts=True))
print(unique)   # [0. 1.]
print(counts)   # [15000 15000]



# y = np.reshape(y, (-1,1)) 
# print(y.shape)  # (19997, 1)

# ohe = OneHotEncoder(sparse = False)
# y_ohe = ohe.fit_transform(y)
# print(y_ohe.shape)  # (19997, 2)






#2. 모델
model = Sequential()
model.add(Conv2D(44, (2,2), padding='same', strides=2, input_shape = (80,80,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), padding='same', activation='relu' ))
model.add(Conv2D(22, (2,2), padding='same',  strides=2, activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='relu')) 
model.add(Dense(2, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc',
                mode='min',
                patience=20,
                verbose=1,
                restore_best_weights=True
                )
# print(x_train.shape, y_train.shape)
model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.2)

#4 평가, 예측
results = model.evaluate(x_train, y_train)
print('loss', results[0])
print('acc', results[1])
# print("걸린시간 : ", round(end - start, 2),"초")

'''


# loss 0.6632859706878662
# acc 0.8082143068313599


loss 0.560442328453064
acc 0.6305333375930786


'''

