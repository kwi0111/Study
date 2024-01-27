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
path_train = 'c:/_data/image/rps/'
xy_train = train_datagen.flow_from_directory(  
# DirectoryIterator 여기서 x는 (배치 크기, *표적 크기, 채널)의 형태의 이미지 배치로 구성된 numpy 배열이고 y는 그에 대응하는 라벨로 이루어진 numpy 배열
    path_train,
    target_size=(150, 150),     
    batch_size=20000,         # 몇장씩 수치화 할거냐     
    class_mode='categorical',
    # color_mode='grayscale',
    shuffle=True,
)   # Found 160 images belonging to 2 classes.

x_train = xy_train[0][0]    # 
y_train = xy_train[0][1]    # 



x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    train_size=0.9,
                                                    random_state=2024,
                                                    shuffle=True,
                                                    stratify=y_train,
                                                    )


print(x_train.shape, y_train.shape) #2268
# test_datagen = ImageDataGenerator(
#     rescale=1./255, 
# )

# xy_test = test_datagen.flow_from_directory(
#     path_test,
#     target_size=(200, 200), 
#     batch_size=5000,
#     class_mode='categorical',
#     # color_mode='grayscale',
#     shuffle=True,
# )


augumet_size = 100      # 가상 이미지의 수 (변수)

randidx = np.random.randint(x_train.shape[0], size=augumet_size)    # 중에서 2000개의 숫자를 뽑아내라.
                                                                    # np.random.randint(60000, 40000)

# print(randidx)  # [ 209  511 1544 ...   32 3048 2190]
# print(np.min(randidx), np.max(randidx)) # 0 3308

x_augumented = x_train[randidx].copy()  # 메모리 원데이터에 영향을 미치지 않기위해서 사용
y_augumented = y_train[randidx].copy()   # 키벨류로 쌍이 맞음
# print(x_augumented)
# print(x_augumented.shape)   # (8000, 200, 200, 3)
# print(y_augumented)
# print(y_augumented.shape)   # (8000,)

# x_augumented = x_augumented.reshape(-1, 28, 28, 1)

# x_augumented = x_augumented.reshape(
#     x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augumet_size,
    shuffle=False,
    
).next()[0]    #  개 변환

# print(x_augumented)
# print(x_augumented.shape)   # (8000, 200, 200, 3)

# print(x_train.shape)    #(3309, 200, 200, 3)
# x_train = x_train.reshape(60000, 28, 28, 1)   
# x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented))        # 사슬 같이 잇다
y_train = np.concatenate((y_train, y_augumented))        # 사슬 같이 잇다

print(x_train.shape, y_train.shape) # (11309, 200, 200, 3) (11309,)
print(x_test.shape, y_test.shape)   # (3, 200, 200, 3) (3,)


np_path = 'c:/_data/_save_npy/'
np.save(np_path + 'keras42_10_x_train.npy', arr=x_train)    
np.save(np_path + 'keras42_10_y_train.npy', arr=y_train)   
np.save(np_path + 'keras42_10_x_test.npy', arr=x_test)    
np.save(np_path + 'keras42_10_y_test.npy', arr=y_test)    

'''
'''
'''
'''
'''
'''
